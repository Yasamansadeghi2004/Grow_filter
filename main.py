# yasaman sadeghi
import cv2
import numpy as np

# تصویر را بارگیری میکند
img = cv2.imread('person.png')

# به تصویر خاکستری تبدیل میکند
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# تصویر ورودی را به عنوان ماسک آستانه‌ای تبدیل میکند
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

# ماسک را منفی میکند
mask = 255 - mask

# اعمال مورفولوژی برای حذف نویزهای اضافی منفرد
# از borderconstant سیاه استفاده میکند زیرا پیش‌زمینه به لبه‌ها می‌رسد
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# کانال آلفا را مات میکند
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

#  خطی میکشد به طوری که از 127.5 به 0 برود، اما 255 باقی میماند
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# ماسک را وارد کانال آلفا میکند
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# تصویر ماسک شده نتیجه را ذخیره میکند
cv2.imwrite('person_transp_bckgrnd.png', result)

# نتیجه را نمایش میدهد اما  شفافیت نشان داده نمی‌شود
cv2.imshow("INPUT", img)
cv2.imshow("GRAY", gray)
cv2.imshow("MASK", mask)
cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()