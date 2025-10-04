from PIL import Image, ImageDraw, ImageFont
import io
import os
import re
from typing import Tuple, Dict, List

class ImageGeneratorService:
    def __init__(self):
        # 字体文件在项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.font_path = os.path.join(project_root, '阿里巴巴普惠体.otf')
        
        # 检查字体文件是否存在
        if not os.path.exists(self.font_path):
            print(f"未找到字体文件: {self.font_path}")
            # 使用默认字体作为备用
            self.font_path = None
        else:
            print(f"使用字体: {self.font_path}")
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色转换为RGB元组"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _calculate_text_height(self, text: str, font: ImageFont.FreeTypeFont, 
                             width: int, margin: int) -> int:
        """计算文本所需的实际高度"""
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        
        # 清理文本，处理换行符 - 支持多种换行符格式
        # 确保转义的换行符被正确处理
        text = text.replace('\\n', '\n')
        
        # 分割文本行
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if not line.strip():
                # 空行也要保留，以实现换行效果
                wrapped_lines.append('')
                continue
                
            # 基于实际文字宽度进行换行判断
            available_width = width - 2 * margin
            
            # 检查整行是否能放下
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]
            
            if line_width <= available_width:
                # 整行能放下，不需要换行
                wrapped_lines.append(line)
            else:
                # 需要换行，逐字符累计宽度来分割
                current_line = ""
                for char in line:
                    test_line = current_line + char
                    test_bbox = font.getbbox(test_line)
                    test_width = test_bbox[2] - test_bbox[0]
                    
                    if test_width <= available_width:
                        current_line = test_line
                    else:
                        # 当前行已满，开始新行
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = char
                
                # 添加最后一行
                if current_line:
                    wrapped_lines.append(current_line)
        
        # 计算总行高
        line_height = font.getbbox('中')[3] + 5  # 行间距
        total_height = len(wrapped_lines) * line_height
        
        return total_height + 2 * margin
    
    def generate(self, text: str, width: int = 800, font_size: int = 22, 
                margin: int = 40, text_color: str = '#2D3748', bg_color: str = '#FFFFFF',
                format: str = 'png', quality: int = 95,
                quick_preview: bool = False) -> Tuple[io.BytesIO, int]:
        """
        生成文本图片
        
        Args:
            text: 文本内容
            width: 图片宽度
            font_size: 字体大小
            margin: 边距
            text_color: 文字颜色（十六进制）
            bg_color: 背景颜色（十六进制）
            format: 图片格式 (png, jpeg, webp, bmp)
            quality: 图片质量 (1-100，仅对jpeg有效)
            quick_preview: 是否为快速预览模式
            
        Returns:
            (图片数据, 实际高度) 元组
        """
        
        # 获取字体
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            # 如果字体加载失败，使用默认字体
            font = ImageFont.load_default()
        
        # 计算所需高度
        if quick_preview and len(text) > 200:
            # 快速预览模式下截断长文本
            text = text[:200] + '...'
        
        text_height = self._calculate_text_height(text, font, width, margin)
        height = text_height
        
        # 创建图片
        image = Image.new('RGB', (width, height), self._hex_to_rgb(bg_color))
        draw = ImageDraw.Draw(image)
        
        # 处理文本换行 - 支持多种换行符格式
        # 再次确保转义的换行符被正确处理
        text = text.replace('\\n', '\n')
        
        # 分割文本行
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if not line.strip():
                # 空行也要保留，以实现换行效果
                wrapped_lines.append('')
                continue
                
            # 基于实际文字宽度进行换行判断
            available_width = width - 2 * margin
            
            # 检查整行是否能放下
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]
            
            if line_width <= available_width:
                # 整行能放下，不需要换行
                wrapped_lines.append(line)
            else:
                # 需要换行，逐字符累计宽度来分割
                current_line = ""
                for char in line:
                    test_line = current_line + char
                    test_bbox = font.getbbox(test_line)
                    test_width = test_bbox[2] - test_bbox[0]
                    
                    if test_width <= available_width:
                        current_line = test_line
                    else:
                        # 当前行已满，开始新行
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = char
                
                # 添加最后一行
                if current_line:
                    wrapped_lines.append(current_line)
        
        # 绘制文本
        y_position = margin
        line_height = font.getbbox('中')[3] + 5
        
        for line in wrapped_lines:
            if line.strip():
                # 计算文本宽度以居中显示
                bbox = font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                x_position = margin
                
                # 绘制文字
                draw.text((x_position, y_position), line, 
                         font=font, fill=self._hex_to_rgb(text_color))
            # 即使是空行也要增加行高以实现换行效果
            y_position += line_height
        
        # 转换为指定格式
        output = io.BytesIO()
        
        # 根据格式设置保存参数
        save_params = {
            'format': format.upper()
        }
        
        if format.lower() == 'png':
            save_params['optimize'] = True
        elif format.lower() == 'jpeg':
            save_params['quality'] = quality
            save_params['optimize'] = True
        elif format.lower() == 'webp':
            save_params['quality'] = quality
        elif format.lower() == 'bmp':
            pass  # BMP不需要特殊参数
        
        image.save(output, **save_params)
        output.seek(0)
        
        return output, height
