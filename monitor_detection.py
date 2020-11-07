import cv2
import numpy as np


class Monitor_detection():
    def __call__(self, img):
        height, width ,_ = img.shape
        line_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.Canny(gray_img,30,50)
        mean_img = cv2.blur(gray, ksize=(16,16))
        _, threshold_img = cv2.threshold(mean_img, 12, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts.sort(key=cv2.contourArea, reverse=True)  # 面積が大きい順に並べ替える。
        warp = None

        for i, c in enumerate(cnts):
            arclen = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*arclen, True)
            level = 1 - float(i)/len(cnts)  # 面積順に色を付けたかったのでこんなことをしている。

            if len(approx) == 4:
                cv2.drawContours(line_img, approx, -1, (0, 0, 255*level), 2)
                if warp is None:
                    warp = approx.copy()  # 一番面積の大きな四角形をwarpに保存。
            else:
                cv2.drawContours(line_img, approx, -1, (0, 255*level, 0), 2)
            
            for pos in approx:
                    cv2.circle(line_img, tuple(pos[0]), 4, (255*level, 0, 0))
        
        if warp is not None:
            result = self.transform_by4(img, warp[:,0,:])  # warpが存在した場合、そこだけくり抜いたものを作る。
            if (result.shape[0]  < height*0.65) or (result.shape[1] < width*0.65):
                result = img
        
        return result
            
        
    def transform_by4(self, img, points):
        points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
        top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
        bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
        points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。

        width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
        height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))

        dst = np.array([
                np.array([0, 0]),
                np.array([width-1, 0]),
                np.array([width-1, height-1]),
                np.array([0, height-1]),
                ], np.float32)

        trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
        return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く