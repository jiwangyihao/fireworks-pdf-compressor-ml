"""
Complete Document Enhancement Pipeline
SLBR QR Detection + Real-ESRGAN Enhancement + DoxaPy Post-processing

流程:
1. 运行 SLBR 获取水印遮罩
2. 从遮罩中定位 QR 码区域
3. 用 Real-ESRGAN 增强（保护 QR 区域）
4. 应用 DoxaPy 白化增强
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== Utility Functions ==============

def cv2_imread_chinese(path):
    """Read image with Chinese path"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def cv2_imwrite_chinese(path, img):
    """Write image with Chinese path"""
    _, ext = os.path.splitext(str(path))
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
        return True
    return False

# ============== SLBR Model (Simplified for QR Detection) ==============

SLBR_REPO = r"C:\Users\34404\Documents\GitHub\SLBR-Visible-Watermark-Removal"
sys.path.insert(0, SLBR_REPO)

from src.networks.blocks import UpConv, DownConv, MBEBlock, SMRBlock, CFFBlock, ResDownNew, ResUpNew, ECABlock
from argparse import Namespace

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)

class CoarseEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, blocks=1, start_filters=32, residual=True, norm=nn.BatchNorm2d, act=F.relu):
        super(CoarseEncoder, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs

class SharedBottleNeck(nn.Module):
    def __init__(self, in_channels=512, depth=5, shared_depth=2, start_filters=32, blocks=1, residual=True,
                 concat=True, norm=nn.BatchNorm2d, act=F.relu, dilations=[1,2,5]):
        super(SharedBottleNeck, self).__init__()
        self.down_convs = []
        self.up_convs = []
        self.up_im_atts = []
        self.up_mask_atts = []

        dilations = [1,2,5]
        start_depth = depth - shared_depth
        max_filters = 512
        outs = in_channels
        for i in range(start_depth, depth):
            ins = in_channels if i == start_depth else outs
            outs = min(ins * 2, max_filters)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act, dilations=dilations)
            self.down_convs.append(down_conv)

            if i < depth - 1:
                up_conv = UpConv(min(outs*2, max_filters), outs, blocks, residual=residual, concat=concat, norm=norm, act=F.relu, dilations=dilations)
                self.up_convs.append(up_conv)
                self.up_im_atts.append(ECABlock(outs))
                self.up_mask_atts.append(ECABlock(outs))
       
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.up_im_atts = nn.ModuleList(self.up_im_atts)
        self.up_mask_atts = nn.ModuleList(self.up_mask_atts)
        reset_params(self)

    def forward(self, input):
        im_encoder_outs = []
        mask_encoder_outs = []
        x = input
        for i, d_conv in enumerate(self.down_convs):
            x, before_pool = d_conv(x)
            im_encoder_outs.append(before_pool)
            mask_encoder_outs.append(before_pool)
        x_im = x
        x_mask = x

        x = x_im
        for i, nets in enumerate(zip(self.up_convs, self.up_im_atts)):
            up_conv, attn = nets
            before_pool = im_encoder_outs[-(i+2)] if im_encoder_outs else None
            x = up_conv(x, before_pool, se=attn)
        x_im = x

        x = x_mask       
        for i, nets in enumerate(zip(self.up_convs, self.up_mask_atts)):
            up_conv, attn = nets
            before_pool = mask_encoder_outs[-(i+2)] if mask_encoder_outs else None
            x = up_conv(x, before_pool, se=attn)
        x_mask = x

        return x_im, x_mask

class CoarseDecoder(nn.Module):
    def __init__(self, args, in_channels=512, out_channels=3, norm='bn', act=F.relu, depth=5, blocks=1, residual=True,
                 concat=True, use_att=False):
        super(CoarseDecoder, self).__init__()
        self.up_convs_bg = []
        self.up_convs_mask = []
        self.atts_bg = []
        self.atts_mask = []
        self.use_att = use_att
        outs = in_channels
        for i in range(depth): 
            ins = outs
            outs = ins // 2
            up_conv = MBEBlock(args.bg_mode, ins, outs, blocks=blocks, residual=residual, concat=concat, norm='in', act=act)
            self.up_convs_bg.append(up_conv)
            if self.use_att:
                self.atts_bg.append(ECABlock(outs))
            
            up_conv = SMRBlock(args, ins, outs, blocks=blocks, residual=residual, concat=concat, norm=norm, act=act)
            self.up_convs_mask.append(up_conv)
            if self.use_att:
                self.atts_mask.append(ECABlock(outs))
        self.conv_final_bg = nn.Conv2d(outs, out_channels, 1,1,0)
        
        self.up_convs_bg = nn.ModuleList(self.up_convs_bg)
        self.atts_bg = nn.ModuleList(self.atts_bg)
        self.up_convs_mask = nn.ModuleList(self.up_convs_mask)
        self.atts_mask = nn.ModuleList(self.atts_mask)
        reset_params(self)

    def forward(self, bg, fg, mask, encoder_outs=None):
        bg_x = bg
        mask_x = mask
        mask_outs = []
        bg_outs = []
        for i, up_convs in enumerate(zip(self.up_convs_bg, self.up_convs_mask)):
            up_bg, up_mask = up_convs
            before_pool = encoder_outs[-(i+1)] if encoder_outs else None

            if self.use_att:
                mask_before_pool = self.atts_mask[i](before_pool) if before_pool is not None else None
                bg_before_pool = self.atts_bg[i](before_pool) if before_pool is not None else None
            else:
                mask_before_pool = before_pool
                bg_before_pool = before_pool
                
            smr_outs = up_mask(mask_x, mask_before_pool)
            mask_x = smr_outs['feats'][0]
            primary_map, self_calibrated_map = smr_outs['attn_maps']
            mask_outs.append(primary_map)
            mask_outs.append(self_calibrated_map)

            bg_x = up_bg(bg_x, bg_before_pool, self_calibrated_map.detach())
            bg_outs.append(bg_x)

        if self.conv_final_bg is not None:
            bg_x = self.conv_final_bg(bg_x)
            mask_x = mask_outs[-1]
            bg_outs = [bg_x] + bg_outs
        return bg_outs, [mask_x] + mask_outs, None

class Refinement(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, shared_depth=2, down=ResDownNew, up=ResUpNew, ngf=32, n_cff=3, n_skips=3):
        super(Refinement, self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, ngf, 3,1,1), nn.InstanceNorm2d(ngf), nn.LeakyReLU(0.2))
        self.down1 = down(ngf, ngf)
        self.down2 = down(ngf, ngf*2)
        self.down3 = down(ngf*2, ngf*4, pooling=False, dilation=True)

        self.dec_conv2 = nn.Sequential(nn.Conv2d(ngf*1, ngf*1, 1,1,0))
        self.dec_conv3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*1, 1,1,0), nn.LeakyReLU(0.2), nn.Conv2d(ngf, ngf, 3,1,1), nn.LeakyReLU(0.2))
        self.dec_conv4 = nn.Sequential(nn.Conv2d(ngf*4, ngf*2, 1,1,0), nn.LeakyReLU(0.2), nn.Conv2d(ngf*2, ngf*2, 3,1,1), nn.LeakyReLU(0.2))
        self.n_skips = n_skips

        self.cff_blocks = nn.ModuleList([CFFBlock(ngf=ngf) for _ in range(n_cff)])

        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf + ngf*2 + ngf*4, ngf, 3,1,1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, out_channels, 1,1,0)
        )     
        
    def forward(self, input, coarse_bg, mask, encoder_outs, decoder_outs):
        dec_feat2 = self.dec_conv2(decoder_outs[0]) if self.n_skips >= 1 else 0
        dec_feat3 = self.dec_conv3(decoder_outs[1]) if self.n_skips >= 2 else 0
        dec_feat4 = self.dec_conv4(decoder_outs[2]) if self.n_skips >= 3 else 0

        xin = torch.cat([coarse_bg, mask], dim=1)
        x = self.conv_in(xin)
        
        x, d1 = self.down1(x + dec_feat2)
        x, d2 = self.down2(x + dec_feat3)
        x, d3 = self.down3(x + dec_feat4)

        xs = [d1, d2, d3]
        for block in self.cff_blocks:
            xs = block(xs)

        xs = [F.interpolate(x_hr, size=coarse_bg.shape[2:][::-1], mode='bilinear') for x_hr in xs]
        im = self.out_conv(torch.cat(xs, dim=1))
        return im

class SLBR(nn.Module):
    def __init__(self, args, in_channels=3, depth=5, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=False):
        super(SLBR, self).__init__()
        self.shared = shared_depth = 2
        self.args = args
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)

        self.encoder = CoarseEncoder(in_channels=in_channels, depth=depth - shared_depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn', act=F.relu)
        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4], residual=residual,
                                               concat=concat, norm='in')
        
        self.coarse_decoder = CoarseDecoder(args, in_channels=start_filters * 2 ** (depth - shared_depth),
                                        out_channels=out_channels_image, depth=depth - shared_depth,
                                        blocks=blocks[1], residual=residual, 
                                        concat=concat, norm='bn', use_att=True)

        self.long_skip = long_skip
        
        if args.use_refine:
            self.refinement = Refinement(in_channels=4, out_channels=3, shared_depth=1, n_cff=args.k_refine, n_skips=args.k_skip_stage)
        else:
            self.refinement = None

    def forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        unshared_before_pool = before_pool

        im, mask = self.shared_decoder(image_code)
        ims, mask, wm = self.coarse_decoder(im, None, mask, unshared_before_pool)
        im = ims[0]
        reconstructed_image = torch.tanh(im)
        if self.long_skip:
            reconstructed_image = (reconstructed_image + synthesized).clamp(0, 1)

        reconstructed_mask = mask[0]
        reconstructed_wm = wm
        
        if self.refinement is not None:
            dec_feats = (ims)[1:][::-1]
            coarser = reconstructed_image * reconstructed_mask + (1 - reconstructed_mask) * synthesized
            refine_bg = self.refinement(synthesized, coarser, reconstructed_mask, None, dec_feats)
            refine_bg = (torch.tanh(refine_bg) + synthesized).clamp(0, 1)
            return [refine_bg, reconstructed_image], mask, [reconstructed_wm]
        else:
            return [reconstructed_image], mask, [reconstructed_wm]

# ============== Document Enhancement Pipeline ==============

class DocumentEnhancer:
    def __init__(self, slbr_checkpoint, esrgan_model_path, device='cpu'):
        self.device = device
        self.slbr_model = None
        self.esrgan_session = None
        self.slbr_checkpoint = slbr_checkpoint
        self.esrgan_model_path = esrgan_model_path
        
    def load_slbr(self):
        """Load SLBR model for watermark detection"""
        if self.slbr_model is not None:
            return
            
        args = Namespace(
            bg_mode='res_mask',
            mask_mode='res',
            sim_metric='cos',
            k_center=2,
            project_mode='simple',
            use_refine=True,
            k_refine=3,
            k_skip_stage=3,
        )
        
        self.slbr_model = SLBR(args=args, shared_depth=1, blocks=3, long_skip=True)
        checkpoint = torch.load(self.slbr_checkpoint, map_location=self.device)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        
        self.slbr_model.load_state_dict(new_state_dict, strict=False)
        self.slbr_model.to(self.device)
        self.slbr_model.eval()
        print(f"SLBR loaded: {sum(p.numel() for p in self.slbr_model.parameters())/1e6:.1f}M params")
        
    def load_esrgan(self):
        """Load Real-ESRGAN ONNX model"""
        if self.esrgan_session is not None:
            return
        self.esrgan_session = ort.InferenceSession(
            self.esrgan_model_path, 
            providers=['CPUExecutionProvider']
        )
        print(f"ESRGAN loaded from {self.esrgan_model_path}")
        
    def detect_qr_region(self, image, tile_size=256, overlap=64):
        """Use SLBR to detect QR/watermark regions via tiled processing"""
        self.load_slbr()
        
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Accumulate mask
        mask_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        stride = tile_size - overlap
        
        with torch.no_grad():
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    y_start = max(0, y_end - tile_size)
                    x_start = max(0, x_end - tile_size)
                    
                    tile = img_rgb[y_start:y_end, x_start:x_end]
                    
                    # Pad if necessary
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                    
                    # Convert to tensor
                    tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                    
                    # Run SLBR
                    outputs = self.slbr_model(tile_tensor)
                    mask_tile = outputs[1][0].squeeze(0).squeeze(0).cpu().numpy()
                    
                    # Remove padding
                    if pad_h > 0:
                        mask_tile = mask_tile[:-pad_h]
                    if pad_w > 0:
                        mask_tile = mask_tile[:, :-pad_w]
                    
                    mask_sum[y_start:y_end, x_start:x_end] += mask_tile
                    weight_sum[y_start:y_end, x_start:x_end] += 1
        
        # Average
        mask_avg = mask_sum / np.maximum(weight_sum, 1e-6)
        return mask_avg
    
    def extract_qr_box_from_mask(self, mask, min_area=2000):
        """Extract QR bounding box from SLBR mask"""
        h, w = mask.shape
        
        # Focus on bottom-right quadrant
        bottom_right = mask[h//2:, w//2:]
        
        # Threshold
        _, binary = cv2.threshold((bottom_right * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        qr_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            
            if 0.5 < aspect_ratio < 2.0 and area > min_area:
                # Adjust to full image coordinates
                margin = 30
                x_min = max(0, x + w // 2 - margin)
                y_min = max(0, y + h // 2 - margin)
                x_max = min(w, x + w // 2 + cw + margin)
                y_max = min(h, y + h // 2 + ch + margin)
                qr_boxes.append((x_min, y_min, x_max, y_max))
        
        return qr_boxes
    
    def create_protection_mask(self, shape, qr_boxes, feather=30):
        """Create soft protection mask for QR regions"""
        h, w = shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        for x_min, y_min, x_max, y_max in qr_boxes:
            mask[y_min:y_max, x_min:x_max] = 0
        
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), feather/2)
        
        return mask
    
    def apply_esrgan_tiled(self, image, tile_size=128, overlap=16):
        """Apply ESRGAN with tiled processing"""
        self.load_esrgan()
        
        h, w = image.shape[:2]
        input_name = self.esrgan_session.get_inputs()[0].name
        
        output = np.zeros_like(image, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        
        stride = tile_size - overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                
                # Pad if needed
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                # Prepare input
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tile_input = np.transpose(tile_rgb, (2, 0, 1))[np.newaxis, ...]
                
                # Run inference
                result = self.esrgan_session.run(None, {input_name: tile_input})[0]
                
                result = np.transpose(result[0], (1, 2, 0))
                result = np.clip(result, 0, 1)
                
                # Downsample if upscaled
                if result.shape[0] != tile_size:
                    result = cv2.resize(result, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
                
                result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Remove padding
                if pad_h > 0:
                    result_bgr = result_bgr[:-pad_h]
                if pad_w > 0:
                    result_bgr = result_bgr[:, :-pad_w]
                
                output[y_start:y_end, x_start:x_end] += result_bgr.astype(np.float32)
                weight[y_start:y_end, x_start:x_end] += 1
        
        output = output / np.maximum(weight[:, :, np.newaxis], 1)
        return output.astype(np.uint8)
    
    def apply_doxa_enhancement(self, image, blend_ratio=0.15):
        """Apply DoxaPy binarization with blending"""
        try:
            import doxapy
        except ImportError:
            print("DoxaPy not available, skipping")
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        gray = gray.astype(np.uint8)
        
        # Create output array
        binary = np.zeros_like(gray, dtype=np.uint8)
        
        # Apply Sauvola binarization - correct API
        binarizer = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
        binarizer.initialize(gray)
        binarizer.to_binary(binary)
        
        # Blend with original (keeps gray tones)
        blended = cv2.addWeighted(gray, 1 - blend_ratio, binary, blend_ratio, 0)
        
        # Convert back to BGR if needed
        if len(image.shape) == 3:
            blended = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
        
        return blended
    
    def enhance(self, image, enable_qr_protection=True, enable_esrgan=True, enable_doxa=True, doxa_blend=0.15):
        """
        Complete enhancement pipeline
        
        Args:
            image: Input BGR image
            enable_qr_protection: Detect and protect QR code regions
            enable_esrgan: Apply Real-ESRGAN enhancement
            enable_doxa: Apply DoxaPy whitening
            doxa_blend: Blend ratio for DoxaPy (0-1)
        
        Returns:
            Enhanced image
        """
        result = image.copy()
        qr_boxes = []
        protection_mask = None
        
        # Step 1: Detect QR regions
        if enable_qr_protection:
            print("Detecting QR regions with SLBR...")
            slbr_mask = self.detect_qr_region(image)
            qr_boxes = self.extract_qr_box_from_mask(slbr_mask)
            print(f"Found {len(qr_boxes)} QR regions: {qr_boxes}")
            
            if qr_boxes:
                protection_mask = self.create_protection_mask(image.shape, qr_boxes)
        
        # Step 2: Apply ESRGAN
        if enable_esrgan:
            print("Applying Real-ESRGAN enhancement...")
            enhanced = self.apply_esrgan_tiled(image)
            
            if protection_mask is not None:
                # Blend: enhanced * mask + original * (1 - mask)
                mask_3ch = protection_mask[:, :, np.newaxis]
                result = (enhanced * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
            else:
                result = enhanced
        
        # Step 3: Apply DoxaPy whitening
        if enable_doxa:
            print(f"Applying DoxaPy whitening (blend={doxa_blend})...")
            result = self.apply_doxa_enhancement(result, blend_ratio=doxa_blend)
        
        return result, qr_boxes, protection_mask


def main():
    output_dir = Path(r"C:\Users\34404\Documents\GitHub\fireworks-notes-attachments\校内资源\材料科学与工程学院\材料科学与工程基础")
    test_image_path = output_dir / "test_gray_original.png"
    slbr_checkpoint = Path(SLBR_REPO) / "slbr_model.pth"
    esrgan_model = output_dir / "models" / "real-esrgan-x4plus-128.onnx"
    
    print("Loading test image...")
    image = cv2_imread_chinese(test_image_path)
    if image is None:
        print(f"Cannot load: {test_image_path}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Create enhancer
    enhancer = DocumentEnhancer(
        slbr_checkpoint=str(slbr_checkpoint),
        esrgan_model_path=str(esrgan_model),
        device='cpu'
    )
    
    # Test different configurations
    configs = [
        ("full_pipeline", {"enable_qr_protection": True, "enable_esrgan": True, "enable_doxa": True, "doxa_blend": 0.15}),
        ("esrgan_protected", {"enable_qr_protection": True, "enable_esrgan": True, "enable_doxa": False}),
        ("esrgan_doxa_20", {"enable_qr_protection": True, "enable_esrgan": True, "enable_doxa": True, "doxa_blend": 0.20}),
    ]
    
    for name, config in configs:
        print(f"\n=== Testing: {name} ===")
        result, qr_boxes, mask = enhancer.enhance(image, **config)
        
        output_path = output_dir / f"test_pipeline_{name}.png"
        cv2_imwrite_chinese(str(output_path), result)
        print(f"Saved: {output_path.name}")
        
        # Save visualization with QR boxes
        if qr_boxes:
            vis = result.copy()
            for x_min, y_min, x_max, y_max in qr_boxes:
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2_imwrite_chinese(str(output_dir / f"test_pipeline_{name}_boxes.png"), vis)
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
