from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv


def mainTableDetector(imageName):
    # Load model
    config_file = '/home/keshav/workspace/wisflux/table-detect/table-detection-server/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
    checkpoint_file = '/home/keshav/workspace/wisflux/table-detect/table-detection-server/CascadeTabNet/epoch_24.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Test a single image
    img = "/home/keshav/workspace/wisflux/table-detect/table-detection-server/pdfs/newfile/{}".format(
        imageName)

    # Run Inference
    result = inference_detector(model, img)
    print(result[0])
    # Visualization results
    show_result_pyplot(img, result, ('Bordered', 'cell',
                                     'Borderless'), score_thr=0.85)

    return result[0][0]
