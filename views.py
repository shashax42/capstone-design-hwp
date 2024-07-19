import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings
from django.contrib import messages
import run
# import projector_mod
import pickle
import base64
import io
import logging
from django.shortcuts import render, redirect
from .models import Img, CameraImage
from django.http import HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.db import connection
from django.http import JsonResponse


logger = logging.getLogger(__name__)


def guide(request):
    logger.info("LOGGER: Guide")
    # version = get_python_version()
    # logger.info(f"LOGGER: test {version}")
    return render(request, 'guide.html')


@login_required
def upload_img(request):

    if request.method == "POST":
        post = Img()
        post.date = timezone.now()
        post.image = request.FILES['image']
        post.save()
        logger.info("LOGGER: Upload_img successed")

        # 현재 로그인한 사용자의 아이디를 가져오기
        user_id = request.user.username

        # SQL 쿼리 실행
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT user_id FROM styled_images WHERE user_id = %s", [user_id])
            user_id_exists = cursor.fetchone()

            if user_id_exists:  # 이미 user_id가 존재하는 경우
                cursor.execute(
                    "SELECT * FROM auth_user WHERE username = %s", [user_id])
                results = [row[3] for row in cursor.fetchall()]

                logger.info(f"LOGGER: test: {results}")

            else:  # user_id가 존재하지 않는 경우 (user_id를 추가)
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO styled_images (user_id) VALUES (%s)", [user_id])

        return redirect('/emo_create/detail/'+str(post.id), {'post': post})
    else:
        post = Img()
        return render(request, 'upload_img.html', {'post': post})


@csrf_exempt
@login_required
def webcam(request):
    if request.method == 'POST':
        image = request.FILES.get('webcam_image')
        CameraImage.objects.create(image=image)
    images = CameraImage.objects.all()
    context = {
        'images': images
    }
    return render(request, 'webcam.html', context)


def detail(request, pk):
    img = Img.objects.get(id=pk)
    check = face_detection(img.image.path)
    if(check == 'many faces recognized'):
        messages.error(request, '얼굴인식이 2개 이상 존재합니다.')
        return redirect('upload_img')
    elif(check == 'cognitive failure'):
        messages.error(request, '얼굴인식이 되지 않습니다.')
        return redirect('upload_img')

    # 이미지 처리 함수 호출
    processed_image = process_image(img.image.path)

    # 이미지 변환 및 인코딩
    # _, buffer = cv2.imencode('.jpg', processed_image)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')

    # 이미지 저장 경로
    processed_image_path = os.path.join(
        '/media/change_images/resized_image.jpg')

    logger.info(f"LOGGER: Processed_Image_Path: {processed_image_path}")

    # 이미지 저장
    cv2.imwrite(processed_image_path, processed_image)

    if request.method == 'POST':
        selected_style = request.POST.get('radio')
        # 선택한 스타일 값을 기반으로 작업 수행
        # return render(request, 'result.html', {'selected_style': selected_style})
        
        return render(request, 'detail.html', {'img': img, 'processed_image_path': processed_image_path, 'selected_style': selected_style})

    return render(request, 'detail.html', {'img': img, 'processed_image_path': processed_image_path})


def process_image(image_path):
    try:
        # 이미지 로드
        image = cv2.imread(image_path)

        # 그레이 스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 얼굴 인식 분류기 생성
        cascade_path = os.path.join(
            settings.STATICFILES_DIRS[0], 'static/xml/haarcascade_frontalface_alt.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info(f"LOGGER: Cascade_Path: {cascade_path}")

        # 얼굴 인식 적용
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        logger.info(f"LOGGER: Faces: {faces}")

        # 얼굴 개수 확인
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            # 얼굴 중심으로 이미지 확대
            scale = 1.4
            cx = x + w // 2
            cy = y + h // 2
            new_w = int(w * scale)
            new_h = int(h * scale)
            x = cx - new_w // 2
            y = cy - new_h // 2
            x = max(x, 0)
            y = max(y, 0)
            new_w = min(new_w, image.shape[1] - x)
            new_h = min(new_h, image.shape[0] - y)
            cropped = image[y:y+new_h, x:x+new_w]
            resized = cv2.resize(
                cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            # resized_img = cv2.resize(resized, (w, h))
            file_path = os.path.join(
            settings.BASE_DIR, 'media/change_images/resized_image.jpg')

            cv2.imwrite(
                file_path, resized)
            return resized
    except:
        return 'error'

# 테스트
# def process_image(image_path):
#     image = cv2.imread(image_path)

#     # 이미지 처리 로직
#     processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     return processed_image


def process_style(request):
    user_id = request.user.username

    if request.method == 'POST':
        selected_style = request.POST.get('radio')
        # 선택한 스타일에 따라 실행할 pkl 파일 경로 설정
        pkl_path = os.path.join(
            settings.BASE_DIR, 'pkls', f'{selected_style}.pkl')
        img_path = os.path.join(
            settings.BASE_DIR, 'media/change_images/resized_image.jpg')
        logger.info(f"LOGGER: selected_style: {selected_style}")
        logger.info(f"LOGGER: PKL 경로: {pkl_path}")
        if selected_style == 'Face Portrait v1' or selected_style == 'Face Portrait v2' or selected_style == 'Webtoon Face' or selected_style == 'Paprika Animation':
            logger.info(f"LOGGER: make_image 2 실행")
            make_image2(request, img_path)
        else:
            logger.info(f"LOGGER: make_image 1 실행")
            # make_image(pkl_path, img_path, user_id)
            
        img_path = os.path.join(
            settings.BASE_DIR, 'static/proj.png')
        if(selected_style == 'cartoon'):
            img = cv2.imread(img_path)
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, img)

        # pkl 파일 실행
        # with open(pkl_path, 'rb') as f:
        #     model = pickle.load(f)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count FROM styled_images WHERE user_id = %s", [user_id])
            count_tuple = cursor.fetchone()
            count = count_tuple[0]
            img_count = count - 1
        return render(request, 'result.html', {'selected_style': selected_style, 'pkl_path': pkl_path, 'user_id': user_id, 'img_count': img_count})


# def make_image(pkl_path, img_path, user_id):
#     with connection.cursor() as cursor:
#         cursor.execute(
#             "SELECT count FROM styled_images WHERE user_id = %s", [user_id])
#         count_tuple = cursor.fetchone()
#         count = count_tuple[0]
#         save_path='static/{}_proj_{}.png'.format(user_id, count)
#         png_path = '\static/{}_proj_{}.png'.format(user_id, count)
#         count = count + 1
#         with connection.cursor() as cursor:
#             sql = "UPDATE styled_images SET count = %s WHERE user_id = %s"
#             cursor.execute(sql, [count, user_id]) 
#     projector_mod.run_projection(
#         network_pkl=pkl_path,
#         target_fname=img_path,
#         outdir='static',
#         save_video=False,
#         seed=303,
#         num_steps=100,
#     )
#     os.rename('static/proj.png', save_path)
#     png_path_count = count - 1
#     update_styled_image_data(user_id, png_path, png_path_count)

def make_image2(request, img_path):
    #celeba_distill
    #paprika
    #face_paint_512_v1
    #face_paint_512_v2
    user_id = request.user.username
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count FROM styled_images WHERE user_id = %s", [user_id])
        count_tuple = cursor.fetchone()
        count = count_tuple[0]
    logger.info(f"LOGGER: Count: {count}")
    output_path=''
    if request.method == 'POST':
        selected_style = request.POST.get('radio')
        
        if(selected_style == 'Face Portrait v1'):
            run.run_projection(
                mod='face_paint_512_v1',
                input_path=img_path,
                output_path='static/{}_proj_{}.png'.format(user_id, count)
            )
            count = count + 1
            with connection.cursor() as cursor:
                sql = "UPDATE styled_images SET count = %s WHERE user_id = %s"
                cursor.execute(sql, [count, user_id]) 
        elif(selected_style == 'Face Portrait v2'):
            run.run_projection(
                mod='face_paint_512_v2',
                input_path=img_path,
                output_path='static/{}_proj_{}.png'.format(user_id, count)
            )
            count = count + 1
            with connection.cursor() as cursor:
                sql = "UPDATE styled_images SET count = %s WHERE user_id = %s"
                cursor.execute(sql, [count, user_id])
        elif(selected_style == 'Webtoon Face'):
            run.run_projection(
                mod='celeba_distill',
                input_path=img_path,
                output_path='static/{}_proj_{}.png'.format(user_id, count)
            )
            count = count + 1
            with connection.cursor() as cursor:
                sql = "UPDATE styled_images SET count = %s WHERE user_id = %s"
                cursor.execute(sql, [count, user_id])
        elif(selected_style == 'Paprika Animation'):
            run.run_projection(
                mod='paprika',
                input_path=img_path,
                output_path='static/{}_proj_{}.png'.format(user_id, count)
            )
            count = count + 1
            with connection.cursor() as cursor:
                sql = "UPDATE styled_images SET count = %s WHERE user_id = %s"
                cursor.execute(sql, [count, user_id])
    png_path_count = count - 1
    png_path=f'\static/{user_id}_proj_{png_path_count}.png'
    update_styled_image_data(user_id, png_path, png_path_count)

def face_detection(image_path):
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        # 그레이 스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # 얼굴 인식 분류기 생성
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # 얼굴 인식 적용
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # 얼굴 개수 확인
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return 'success'
        elif len(faces) > 1:
            return 'many faces recognized'
        else:
            return 'cognitive failure'
    except:
        return 'error'


# 이미지 다운로드
def download_image(request):
    user_id = request.user.username
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count FROM styled_images WHERE user_id = %s", [user_id])
        count_tuple = cursor.fetchone()
        count = count_tuple[0]
        count = count - 1
    processed_image_path = os.path.join(
        settings.BASE_DIR, f'static/{user_id}_proj_{count}.png')
    with open(processed_image_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type="image/jpeg")
        response['Content-Disposition'] = f'attachment; filename="{user_id}_create_img.jpg"'
        return response
    return render(request, 'detail.html', {'img': img, 'processed_image_path': processed_image_path})

def loading(request):
    return render(request, 'loading.html')


def update_styled_image_data(user_id, png_path, png_path_count):
    
    png_path_count_1 = png_path_count % 5
    if png_path_count == 1 or png_path_count_1 == 1:
        with connection.cursor() as cursor:
            sql = "UPDATE styled_images SET styled_image_path1 = %s WHERE user_id = %s"
            cursor.execute(sql, [png_path, user_id])
    elif png_path_count == 2 or png_path_count_1 == 2:
        with connection.cursor() as cursor:
            sql = "UPDATE styled_images SET styled_image_path2 = %s WHERE user_id = %s"
            cursor.execute(sql, [png_path, user_id])
    elif png_path_count == 3 or png_path_count_1 == 3:
        with connection.cursor() as cursor:
            sql = "UPDATE styled_images SET styled_image_path3 = %s WHERE user_id = %s"
            cursor.execute(sql, [png_path, user_id])
    elif png_path_count == 4 or png_path_count_1 == 4:
        with connection.cursor() as cursor:
            sql = "UPDATE styled_images SET styled_image_path4 = %s WHERE user_id = %s"
            cursor.execute(sql, [png_path, user_id])
    else:
        with connection.cursor() as cursor:
            logger.info(f"LOGGER: 마이 이모티콘꺼: {png_path}")
            sql = "UPDATE styled_images SET styled_image_path0 = %s WHERE user_id = %s"
            cursor.execute(sql, [png_path, user_id])

# 글자 추가
def add_text_to_image(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        color = request.POST.get('radio')
    user_id = request.user.username
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count FROM styled_images WHERE user_id = %s", [user_id])
        count_tuple = cursor.fetchone()
        count = count_tuple[0]
        img_count = count - 1
    img_path = f'static/{ user_id }_proj_{ img_count }.png'
    img = Image.open(img_path)
    w, h = img.size
    font_path = os.path.join(
            settings.BASE_DIR, 'ttfs', 'NanumSquareB.ttf')
    
    logger.info(f"LOGGER: 결과에서 텍스트 추가 text: {text}")
    logger.info(f"LOGGER: 결과에서 텍스트 추가 img_path: {img_path}")
    logger.info(f"LOGGER: 결과에서 텍스트 추가 font_path: {font_path}")

    fontsize = ((w + h) //2 ) // 6
    if len(text) >= 6:
        text = text[:6]
    elif len(text) == 2:
        fontsize = ((w + h) //2 ) // 5
        text = text[0] + " " * 7 + text[1]
    elif len(text) == 3:
        fontsize = ((w + h) //2 ) // 5
        text = text[0] + " " * 2 + text[1] + " " * 2 + text[2]
    elif len(text) == 4:
        fontsize = ((w + h) //2 ) // 5
        text = text[0] + " " * 1 + text[1] + " " * 1 + text[2]  + " " * 1 + text[3]
    
    fnt = ImageFont.truetype(font_path, fontsize, encoding="UTF-8")
    if hasattr(fnt, "getsize_multiline"):
        tw, th = fnt.getsize_multiline(text)
    else:
        tw, th = fnt.getsize(text)
    draw = ImageDraw.Draw(img)
    
    text_x = int((w - tw) / 2)
    text_y = h - th - 25

    if color == "r":
        fill_color = "red"
    elif color == "b":
        fill_color = "blue"
    elif color == "g":
        fill_color = "green"
    elif color == "w":
        fill_color = "white"
    elif color == "k":
        fill_color = "black"
    else:
        fill_color = "black" 
    
    draw.text((text_x, text_y), text, font=fnt, fill=fill_color)
    img.save(img_path)
    
    #img.show()
    return render(request, 'result.html', {'user_id': user_id, 'img_count': img_count})