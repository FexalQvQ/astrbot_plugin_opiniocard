""" Opiniocard 立场卡插件
格式：正文内容|立场
"""
import os
import re
import time
import random
import hashlib
import colorsys
import json
from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont

import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api.provider import LLMResponse
from astrbot.api import logger


# 【延迟导入共享服务】
_icm_available = None


def _ensure_icm():
    global _icm_available
    if _icm_available is not None:
        return _icm_available
    try:
        from astrbot_plugin_image_caption_manager.main import (
            register_caption,
            replace_placeholders,
        )
        _icm_available = True
        logger.info("[Opiniocard] Image Caption Manager 已连接")
        return True
    except ImportError:
        _icm_available = False
        logger.warning("[Opiniocard] Image Caption Manager 未加载，多插件冲突可能无法避免")
        return False


STANCE_LIST = ["agree", "disagree", "doubt"]
TRIGGER_PROBABILITY = 0.05
FORCE_KEYWORDS = ["立场卡"]

# H4 S30 V20 -> RGB
_h, _s, _v = 4/360, 30/100, 20/100
_r, _g, _b = colorsys.hsv_to_rgb(_h, _s, _v)
COLOR_TEXT = (int(_r*255), int(_g*255), int(_b*255))

FONT_SIZE = 48
LETTER_SPACING = 5

TEXTBOX_LEFT = 485
TEXTBOX_TOP = 370
TEXTBOX_RIGHT = 1205
TEXTBOX_BOTTOM = 535
TEXTBOX_WIDTH = TEXTBOX_RIGHT - TEXTBOX_LEFT   # 720
TEXTBOX_HEIGHT = TEXTBOX_BOTTOM - TEXTBOX_TOP  # 165

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1200

LINE_HEIGHT = int(FONT_SIZE * 1.4)

# 最大正文长度（字）
MAX_BODY_LENGTH = 30


class OpiniocardPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.plugin_dir = Path(__file__).parent
        self.asset_dir = self.plugin_dir / "asset"
        self.pic_dir = self.asset_dir / "pictures"
        self.font_dir = self.asset_dir / "font"
        self.temp_dir = self.plugin_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.font_path = self.font_dir / "SourceHanSerifCN-SemiBold-7.ttf"
        self._card_sessions: dict = {}
        # 只保留 UMO 缓存
        self._umo_last_card: dict[str, dict] = {}  # umo -> {text, time}
        self._load_font()
        logger.info(f"[Opiniocard] 插件初始化完成，文字色={COLOR_TEXT}, ICM={_icm_available}")

    def _load_font(self):
        try:
            if self.font_path.exists():
                self._font = ImageFont.truetype(str(self.font_path), FONT_SIZE)
                logger.info(f"[Opiniocard] 字体加载成功: {self.font_path}")
            else:
                logger.warning(f"[Opiniocard] 字体不存在，使用默认字体")
                self._font = ImageFont.load_default()
        except Exception as e:
            logger.error(f"[Opiniocard] 字体加载失败: {e}")
            self._font = ImageFont.load_default()

    def _get_bg_path(self, stance: str) -> Path:
        return self.pic_dir / f"opinion_{stance}.png"

    def _check_bg_exists(self, stance: str) -> bool:
        return self._get_bg_path(stance).exists()

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        # 【关键修复】最先执行：替换引用消息中的图片占位符
        # 必须在任何 return 之前，与是否触发立场卡无关
        await self._fix_quoted_image_placeholder(event, req)

        # 原有逻辑保持不变（立场卡生成触发）
        umo = event.unified_msg_origin
        user_text = event.message_str or ""

        force_trigger = any(kw in user_text for kw in FORCE_KEYWORDS)

        if "词卡" in user_text:
            return
        if getattr(event, '_dialocard_marked', False):
            return
        if umo in self._card_sessions and not self._card_sessions[umo].get("consumed", False):
            return
        if umo in self._card_sessions and self._card_sessions[umo].get("consumed", False):
            del self._card_sessions[umo]

        if not force_trigger and random.random() >= TRIGGER_PROBABILITY:
            return

        self._card_sessions[umo] = {
            "consumed": False,
            "force": force_trigger
        }
        event._opiniocard_marked = True

        prompt_addition = f"""
[OPINIOCARD_FORMAT_REQ]
输出格式：正文|立场
规则：1){MAX_BODY_LENGTH}字内分析观点 2)立场选agree/disagree/doubt中的一个
不允许在正文后不带上"|立场"
示例：欸，确实如此...奈叶香的魔法是幻视|agree
[/OPINIOCARD_FORMAT_REQ]"""

        try:
            from astrbot.core.agent.message import TextPart
            req.extra_user_content_parts.append(TextPart(text=prompt_addition))
        except ImportError:
            req.prompt += "\n\n" + prompt_addition

        logger.info(f"[Opiniocard] {'[强制]' if force_trigger else '[概率]'}已标记: {umo}")

    @filter.on_llm_response(priority=10)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        umo = event.unified_msg_origin
        session_info = self._card_sessions.get(umo)
        if not session_info or session_info.get("consumed", False):
            return

        # 【关键】跳过 chunk 模式
        if getattr(resp, 'is_chunk', False):
            return

        text = resp.completion_text
        if not text or not text.strip():
            return

        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            logger.debug("[Opiniocard] JSON响应，跳过")
            return

        # 检测到词卡高亮标记 {{}}，跳过让 DialoCard 处理
        if re.search(r'\{\{[^{}]+\}\}', text):
            logger.info(f"[Opiniocard] 检测到词卡高亮标记，跳过处理")
            self._cleanup_session(umo)
            return

        # 检测到表情包格式 &&emotion&&，跳过让 emoji_sender 处理
        if re.search(r'&&[a-zA-Z_]+&&', text):
            logger.info(f"[Opiniocard] 检测到表情包标记，跳过处理")
            self._cleanup_session(umo)
            return

        logger.info(f"[Opiniocard] LLM原始输出: {repr(text)}")
        original_text = text  # 【新增】保留原始输出，fallback 时用于找回 AT 标记
        try:
            stance, clean_text = self._parse_card_text(text)
            logger.info(f"[Opiniocard] 解析: stance={stance}, text={repr(clean_text)}")

            # 【修改】未解析到有效立场（含正文超过 MAX_BODY_LENGTH）时，
            # fallback 到文字输出，而不是截断画进图片
            if not stance:
                logger.info(
                    f"[Opiniocard] 未检测到有效立场（含字数超限），fallback 到文字输出: {clean_text}"
                )
                self._fallback_to_text(resp, clean_text, original_text)
                # 不设置 event._opiniocard_image_path，
                # on_decorating_result 检测不到图片路径，自然按纯文本发出
                return

            session_info["consumed"] = True

            if not self._check_bg_exists(stance):
                logger.error(f"[Opiniocard] 底图不存在: {stance}，fallback 到文字输出")
                # 【新增】同样回退纯文字，避免发出脏的 "|agree" 后缀
                self._fallback_to_text(resp, clean_text, original_text)
                return

            image_path = self._generate_card(stance, clean_text)
            if not image_path or not os.path.exists(image_path):
                logger.error("[Opiniocard] 生成失败，fallback 到文字输出")
                # 【新增】同样回退纯文字
                self._fallback_to_text(resp, clean_text, original_text)
                return

            # 【关键修改】构建图片描述，替换 resp.completion_text
            stance_desc_map = {
                "agree": "赞同",
                "disagree": "反驳",
                "doubt": "疑问",
            }
            stance_desc = stance_desc_map.get(stance, "")
            caption = f'[你发送的图片内容：图片中你{stance_desc}说"{clean_text}"]'
            # 替换 LLM 原始输出为图片描述，让框架记录到历史
            resp.completion_text = caption

            # 缓存图片路径供 on_decorating_result 使用
            event._opiniocard_image_path = image_path
            event._opiniocard_text = clean_text
            event._opiniocard_stance = stance

            # 注册到共享服务（供其他插件引用时使用）
            if _ensure_icm():
                from astrbot_plugin_image_caption_manager.main import register_caption
                register_caption(image_path, caption)

            logger.info(f"[Opiniocard] 图片描述已设置: {caption}")

        except Exception as e:
            logger.error(f"[Opiniocard] 异常: {e}", exc_info=True)
        finally:
            self._cleanup_session(umo)

    def _fallback_to_text(self, resp: LLMResponse, clean_text: str, original_text: str):
        """【新增】超字数/无立场/底图缺失/渲染失败时，回退为纯文字输出。

        写回的是经过 _parse_card_text 正则过滤环节的干净正文，
        并从原始输出中找回 AT 标记，保证 @ 功能不丢失。
        """
        fallback_text = (clean_text or "").strip() or (original_text or "").strip()
        at_markers = re.findall(r"\[\s*AT\s*:\s*\d+\s*\]", original_text or "")
        if at_markers:
            resp.completion_text = fallback_text + " " + " ".join(at_markers)
        else:
            resp.completion_text = fallback_text

    def _cleanup_session(self, umo: str):
        if umo in self._card_sessions:
            del self._card_sessions[umo]

    def _parse_card_text(self, text: str) -> Tuple[str, str]:
        """
        解析立场卡文本，支持多种格式变体。
        策略：
        1. 【新增】清理 AT 标记（与 DialoCard 对齐，避免标记混入正文/字数统计）
        2. 从文本末尾反向查找 stance 标记
        3. 支持格式：|stance、||stance、||stance||、&&stance&&（误写容错）
        4. 正文为 stance 之前的全部内容
        5. 连续换行替换为单个空格
        6. 【修改】正文超过 MAX_BODY_LENGTH 时不截断，
           返回空 stance，由 on_llm_response 触发 fallback 到文字输出
        """
        text = text.strip()

        # 【新增】第零步：清理艾特标记
        text = re.sub(r"\[\s*AT\s*:\s*\d+\s*\]", "", text).strip()

        # 第一步：从末尾提取 stance
        stance = ""
        body = text

        # 尝试匹配各种 stance 格式（从末尾开始）
        # 模式1: |agree、||agree、| agree |、||agree|| 等
        # 模式2: &&agree&&（模型误写表情包格式）
        stance_patterns = [
            # ||agree||、|| agree ||、|agree| 等
            (r'[\|&]+\s*([a-zA-Z_]+)\s*[\|&]+\s*$', 1),
            # agree（如果最后一行只有一个单词且是 stance）
            (r'\n\s*([a-zA-Z_]+)\s*$', 1),
        ]
        for pattern, group_idx in stance_patterns:
            match = re.search(pattern, text)
            if match:
                potential_stance = match.group(group_idx).strip().lower()
                if potential_stance in STANCE_LIST:
                    stance = potential_stance
                    body = text[:match.start()].strip()
                    break

        # fallback：如果没匹配到，尝试 rsplit（处理简单 |stance 格式）
        if not stance and "|" in text:
            # 从右往左找最后一个 |
            parts = text.rsplit("|", 1)
            if len(parts) == 2:
                body = parts[0].strip()
                stance = parts[1].strip().lower()
                if stance not in STANCE_LIST:
                    stance = ""

        # 第二步：清理正文（正则过滤环节，fallback 时输出的就是这份干净文本）
        if body:
            # 去掉末尾残留的 |、&、空格
            body = re.sub(r'[\|&\s]+$', '', body).strip()
            # 连续换行（2个及以上）替换为单个空格
            body = re.sub(r'\n{2,}', ' ', body)
            # 单个换行也替换为空格（让多行变一行）
            body = re.sub(r'\n', ' ', body)
            # 连续空格压缩
            body = re.sub(r'\s+', ' ', body).strip()
            # 连续标点规范化
            body = re.sub(r'[，,]+', '，', body)
            body = re.sub(r'[。.]+', '。', body)

        # 【修改】第三步：长度限制 —— 超限时不截断，
        # 返回空 stance 让上层 fallback 到文字输出（与 DialoCard 行为一致）
        if len(body) > MAX_BODY_LENGTH:
            return "", body

        return stance, body

    def _generate_card(self, stance: str, text: str) -> Optional[str]:
        try:
            bg_path = self._get_bg_path(stance)
            bg = Image.open(str(bg_path)).convert("RGBA")
            if bg.size != (CANVAS_WIDTH, CANVAS_HEIGHT):
                bg = bg.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.LANCZOS)

            canvas = bg.copy()
            self._draw_text(canvas, text)

            text_hash = hashlib.md5(f"{stance}_{text}".encode()).hexdigest()[:12]
            output_path = self.temp_dir / f"opiniocard_{stance}_{text_hash}.png"
            canvas.save(str(output_path), "PNG")
            return str(output_path)
        except Exception as e:
            logger.error(f"[Opiniocard] 渲染失败: {e}", exc_info=True)
            return None

    def _draw_text(self, canvas: Image.Image, text: str):
        draw = ImageDraw.Draw(canvas)

        char_widths = []
        for char in text:
            bbox = self._font.getbbox(char)
            w = bbox[2] - bbox[0] if bbox else FONT_SIZE
            char_widths.append(w)

        lines = []
        current_line = []
        current_width = 0

        for i, char in enumerate(text):
            char_w = char_widths[i]
            add_width = char_w if not current_line else char_w + LETTER_SPACING

            if current_line and current_width + add_width > TEXTBOX_WIDTH:
                lines.append(current_line)
                current_line = [i]
                current_width = char_w
            else:
                current_line.append(i)
                current_width += add_width

        if current_line:
            lines.append(current_line)

        total_height = len(lines) * LINE_HEIGHT
        start_y = TEXTBOX_TOP + (TEXTBOX_HEIGHT - total_height) / 2

        for line_idx, line_indices in enumerate(lines):
            line_width = 0
            for j, idx in enumerate(line_indices):
                line_width += char_widths[idx]
                if j < len(line_indices) - 1:
                    line_width += LETTER_SPACING

            x = TEXTBOX_LEFT + (TEXTBOX_WIDTH - line_width) / 2
            y = start_y + line_idx * LINE_HEIGHT

            for idx in line_indices:
                char = text[idx]
                draw.text((x, y), char, font=self._font, fill=COLOR_TEXT)
                x += char_widths[idx] + LETTER_SPACING

    # ============ 【关键修改】使用共享服务替换占位符 ============
    async def _fix_quoted_image_placeholder(self, event: AstrMessageEvent, req):
        """使用共享服务替换引用消息中的图片占位符"""
        if not _ensure_icm():
            return

        # 1. 替换 req.prompt
        if req.prompt:
            new_prompt = replace_placeholders(req.prompt)
            if new_prompt != req.prompt:
                req.prompt = new_prompt
                logger.info(f"[Opiniocard] 已替换 prompt 中的占位符")

        # 2. 替换 extra_user_content_parts
        if req.extra_user_content_parts:
            for part in req.extra_user_content_parts:
                if hasattr(part, 'text'):
                    new_text = replace_placeholders(part.text)
                    if new_text != part.text:
                        part.text = new_text
                        logger.info(f"[Opiniocard] 已替换 extra_part 中的占位符")

        # 3. 替换 contexts 中的历史消息
        if req.contexts:
            for ctx in req.contexts:
                content = ctx.get("content", "")
                if isinstance(content, str):
                    new_content = replace_placeholders(content)
                    if new_content != content:
                        ctx["content"] = new_content
                        logger.info(f"[Opiniocard] 已替换 contexts 中的占位符")
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            new_text = replace_placeholders(text)
                            if new_text != text:
                                part["text"] = new_text
                                logger.info(f"[Opiniocard] 已替换 contexts list 中的占位符")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        image_path = getattr(event, '_opiniocard_image_path', None)
        if not image_path or not os.path.exists(image_path):
            # 【说明】fallback 分支不会设置 _opiniocard_image_path，
            # 走到这里直接放行，纯文本消息原样发出
            return

        try:
            result = event.get_result()
            if result is None:
                return

            # 【关键修改】将 result.chain 中的 Plain 替换为 Image
            # 保留其他插件可能添加的组件（如 Reply、At 等）
            new_chain = []
            for comp in result.chain:
                comp_type = getattr(comp, 'type', None) or \
                            getattr(comp, 'component_type', None) or \
                            comp.__class__.__name__
                if comp_type in ('Plain', 'plain'):
                    # 跳过文本组件（这是图片描述，用户不需要看到）
                    continue
                new_chain.append(comp)

            # 添加图片组件
            new_chain.append(Comp.Image.fromFileSystem(image_path))
            result.chain = new_chain

            # 缓存最近描述（供引用时使用）
            card_text = getattr(event, '_opiniocard_text', None)
            if card_text:
                self._umo_last_card[event.unified_msg_origin] = {
                    "text": card_text,
                    "time": time.time()
                }
                if len(self._umo_last_card) > 50:
                    oldest_key = next(iter(self._umo_last_card))
                    del self._umo_last_card[oldest_key]

            logger.info(f"[Opiniocard] 已装饰结果链，替换为图片: {image_path}")

            # 清理 event 属性
            if hasattr(event, '_opiniocard_image_path'):
                delattr(event, '_opiniocard_image_path')
            if hasattr(event, '_opiniocard_text'):
                delattr(event, '_opiniocard_text')
            if hasattr(event, '_opiniocard_stance'):
                delattr(event, '_opiniocard_stance')

        except Exception as e:
            logger.error(f"[Opiniocard] 消息装饰失败: {e}", exc_info=True)

    async def terminate(self):
        try:
            for f in self.temp_dir.glob("opiniocard_*.png"):
                f.unlink(missing_ok=True)
            self._card_sessions.clear()
            self._umo_last_card.clear()
            logger.info("[Opiniocard] 临时文件和会话已清理")
        except Exception as e:
            logger.error(f"[Opiniocard] 清理失败: {e}")
