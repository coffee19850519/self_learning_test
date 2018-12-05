import cfg
from network import East
from predict import predict
import cv2


def main(img_path):


  east = East()
  east_detect = east.east_network()
  east_detect.load_weights(cfg.saved_model_weights_file_path)
  for i in range(int(cfg.max_self_iteration)):
      predict(east_detect, img_path, i,quiet= True)
      #load predicted images and results
      with open(img_path[:-4] +'.txt') as result_fp:
        results = result_fp.readlines()
      if i != 0:
        image_path = img_path[:-4] + '_' + str(i-1) + '.png'
      else:
        image_path = img_path
      input_img = cv2.imread(image_path)
      ori_img = cv2.imread(img_path)
      #with Image.open(image_path) as im:
        #im_array = im.img_to_array()
       # draw = ImageDraw.Draw(im.convert('RGB'))

      for line in results:
        geo = line.split(',')
        # erase detective text
        cv2.rectangle(input_img,(int((geo[0])) - cfg.erase_offset_pixels, int((geo[1]))- cfg.erase_offset_pixels),
                          (int((geo[4]))+ cfg.erase_offset_pixels, int((geo[5]))+ cfg.erase_offset_pixels),
                        (255, 255, 255), thickness= -1)
        #marked predict results
        cv2.rectangle(ori_img, (int((geo[0])), int((geo[1]))),
                      (int((geo[4])), int((geo[5]))),
                      (255, 0, 0), thickness=1)

      cv2.imwrite(img_path[:-4] + '_' + str(i) + '.png',input_img)
      cv2.imwrite(img_path[:-4] + '_predict_' + str(i) + '.png', ori_img)
        #predict(east_detect, img_path, i)
        #mark predict score
        #marked_im = im.copy()



if __name__ == '__main__':
    target_figure_path = r'C:\Users\LSC-110\Desktop\self_learning_test' \
                         r'\erase_figures\hsa00010.png'
    main(target_figure_path)
