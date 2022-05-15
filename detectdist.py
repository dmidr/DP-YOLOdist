# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import configparser
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class Resolution:
    width = 1280
    height = 720


class Stereo:
    def __init__(self):
        calibration_file = "SN24759.conf"  # download_calibration_file(serial_number)
        if calibration_file == "":
            exit(1)
        print("Calibration file found. Loading...")

        image_size = Resolution()
        image_size.width = 1280
        image_size.height = 720
        self.camera_matrix_left, self.camera_matrix_right, self.map_left_x, self.map_left_y, self.map_right_x, self.map_right_y, self.foclen = self.init_calibration(
            calibration_file, image_size)
        self.baseline = 0.120  # m

        window_size = 3
        self.stereoProcessor = cv2.StereoSGBM_create(minDisparity=0,
                                                numDisparities=32,
                                                blockSize=window_size,
                                                P1=0,
                                                P2=3 * window_size ** 2,
                                                # disp12MaxDiff = 0,
                                                # uniquenessRatio = 10,
                                                speckleWindowSize=100,
                                                speckleRange=2
                                                )
        self.stereoProcessor_R = cv2.ximgproc.createRightMatcher(self.stereoProcessor)

        # WLS FILTER Parameters - tune to adjust disparity
        lmbda = 8000
        sigma = 2.1

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereoProcessor)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)

    def distance(self, img_stereo, coords, dist):
        left_right_image = np.split(img_stereo, 2, axis=1)
        left_rect = cv2.remap(left_right_image[0], self.map_left_x, self.map_left_y, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(left_right_image[1], self.map_right_x, self.map_right_y, interpolation=cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = self.stereoProcessor.compute(grayL, grayR).astype(np.float32)/16
        disparity_L = disparity
        disparity_R = self.stereoProcessor_R.compute(grayR, grayL).astype(np.float32)/16
        disparity_filtered = self.wls_filter.filter(disparity_L, grayL, None, disparity_R)

        for i, det in enumerate(coords):
            if disparity_filtered[int(det[1]), int(det[0])] != 0:
                dist[i] = self.baseline * self.foclen / disparity_filtered[int(det[1]), int(det[0])]  # m

        return dist


    def init_calibration(self, calibration_file, image_size):

        cameraMatrix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

        config = configparser.ConfigParser()
        config.read(calibration_file)

        check_data = True
        resolution_str = ''
        if image_size.width == 2208:
            resolution_str = '2K'
        elif image_size.width == 1920:
            resolution_str = 'FHD'
        elif image_size.width == 1280:
            resolution_str = 'HD'
        elif image_size.width == 672:
            resolution_str = 'VGA'
        else:
            resolution_str = 'HD'
            check_data = False

        T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                       float(config['STEREO']['TY_' + resolution_str] if 'TY_' + resolution_str in config[
                           'STEREO'] else 0),
                       float(config['STEREO']['TZ_' + resolution_str] if 'TZ_' + resolution_str in config[
                           'STEREO'] else 0)])

        left_cam_cx = float(
            config['LEFT_CAM_' + resolution_str]['cx'] if 'cx' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_cy = float(
            config['LEFT_CAM_' + resolution_str]['cy'] if 'cy' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_fx = float(
            config['LEFT_CAM_' + resolution_str]['fx'] if 'fx' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_fy = float(
            config['LEFT_CAM_' + resolution_str]['fy'] if 'fy' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_k1 = float(
            config['LEFT_CAM_' + resolution_str]['k1'] if 'k1' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_k2 = float(
            config['LEFT_CAM_' + resolution_str]['k2'] if 'k2' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_p1 = float(
            config['LEFT_CAM_' + resolution_str]['p1'] if 'p1' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_p2 = float(
            config['LEFT_CAM_' + resolution_str]['p2'] if 'p2' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_p3 = float(
            config['LEFT_CAM_' + resolution_str]['p3'] if 'p3' in config['LEFT_CAM_' + resolution_str] else 0)
        left_cam_k3 = float(
            config['LEFT_CAM_' + resolution_str]['k3'] if 'k3' in config['LEFT_CAM_' + resolution_str] else 0)

        right_cam_cx = float(
            config['RIGHT_CAM_' + resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_cy = float(
            config['RIGHT_CAM_' + resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_fx = float(
            config['RIGHT_CAM_' + resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_fy = float(
            config['RIGHT_CAM_' + resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_k1 = float(
            config['RIGHT_CAM_' + resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_k2 = float(
            config['RIGHT_CAM_' + resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_p1 = float(
            config['RIGHT_CAM_' + resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_p2 = float(
            config['RIGHT_CAM_' + resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_p3 = float(
            config['RIGHT_CAM_' + resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_' + resolution_str] else 0)
        right_cam_k3 = float(
            config['RIGHT_CAM_' + resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_' + resolution_str] else 0)

        R_zed = np.array(
            [float(config['STEREO']['RX_' + resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
             float(config['STEREO']['CV_' + resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
             float(config['STEREO']['RZ_' + resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

        R, _ = cv2.Rodrigues(R_zed)
        cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                                      [0, left_cam_fy, left_cam_cy],
                                      [0, 0, 1]])

        cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                                       [0, right_cam_fy, right_cam_cy],
                                       [0, 0, 1]])

        distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

        distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

        T = np.array([[T_[0]], [T_[1]], [T_[2]]])
        R1 = R2 = P1 = P2 = np.array([])

        R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                           cameraMatrix2=cameraMatrix_right,
                                           distCoeffs1=distCoeffs_left,
                                           distCoeffs2=distCoeffs_right,
                                           R=R, T=T,
                                           flags=cv2.CALIB_ZERO_DISPARITY,
                                           alpha=0,
                                           imageSize=(image_size.width, image_size.height),
                                           newImageSize=(image_size.width, image_size.height))[0:4]

        map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1,
                                                             (image_size.width, image_size.height), cv2.CV_32FC1)
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2,
                                                               (image_size.width, image_size.height), cv2.CV_32FC1)

        cameraMatrix_left = P1
        cameraMatrix_right = P2

        return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y, left_cam_fx


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    stereo = Stereo()

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap in dataset:  # !!! im needs to be split in utils.datasets class to be resized correctly
        t1 = time_sync()
        imstereo = im0s  # save stereo images for distance estimation
        if webcam:
            im0s = [np.split(x, 2, axis=1)[0] for x in im0s]
        else:
            im0s = np.split(im0s, 2, axis=1)[0]
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        dt[2] += t4 - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # DISTANCE
        # Get list of bbox coordinates from pred

        if webcam:
            dist = [None] * len(pred)
            for i, x in enumerate(pred):
                bbox_coords = x.clone().detach().cpu().numpy()
                bbox_coords = bbox_coords[:, 0:4]  # remove class, conf
                bbox_coords = scale_coords(im.shape[2:], bbox_coords, im0s[0].shape).round()
                ctr_coords = np.zeros((bbox_coords.shape[0], 2))
                ctr_coords[:, 0] = (bbox_coords[:, 0] + bbox_coords[:, 2]) * 0.5  # x center coords
                ctr_coords[:, 1] = (bbox_coords[:, 1] + bbox_coords[:, 3]) * 0.5  # y center coords
                print(ctr_coords)
                dist[i] = np.zeros((len(bbox_coords), 1))
                dist[i] = stereo.distance(imstereo[i], ctr_coords, dist[i])
                dist[i] = np.flip(dist[i])
        else:
            bbox_coords = pred[0].clone().detach().cpu().numpy()
            bbox_coords = bbox_coords[:, 0:4]  # remove class, conf
            bbox_coords = scale_coords(im.shape[2:], bbox_coords, im0s.shape).round()
            ctr_coords = np.zeros((bbox_coords.shape[0], 2))
            ctr_coords[:, 0] = (bbox_coords[:, 0] + bbox_coords[:, 2]) * 0.5  # x center coords
            ctr_coords[:, 1] = (bbox_coords[:, 1] + bbox_coords[:, 3]) * 0.5  # y center coords
            dist = np.zeros((len(bbox_coords), 1))
            dist = stereo.distance(imstereo, ctr_coords, dist)
            dist = np.flip(dist)
        #return 0

        t5 = time_sync()
        dt[3] += t5 - t4

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                index_dist = 0
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if webcam:
                            line = (cls, *xywh, conf, dist[i][index_dist, 0]) if save_conf else (
                                cls, *xywh, dist[i][index_dist, 0])  # label format
                        else:
                            line = (cls, *xywh, conf, dist[index_dist, 0]) if save_conf else (
                                cls, *xywh, dist[index_dist, 0])  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if webcam:
                            label = None if hide_labels else (
                                f'{names[c]} {(dist[i][index_dist, 0]):.2f}' if hide_conf else f'{names[c]} {conf:.2f} {(dist[i][index_dist, 0]):.2f}')
                        else:
                            label = None if hide_labels else (f'{names[c]} {(dist[index_dist, 0]):.2f}' if hide_conf else f'{names[c]} {conf:.2f} {(dist[index_dist, 0]):.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    index_dist += 1

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. (det: {t3 - t2:.3f} s, dist: {t5-t4:.3f} s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(fps, w, h)
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms depth per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
