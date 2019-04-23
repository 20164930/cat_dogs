import json
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from user import load_model


def index(request):
    return render(request,"index.html")

def imgurl(request):
    if(request.is_ajax()):
        file = request.FILES.get("img")
        if file:  # 如果文件存在
            with open('C:\\Users\\Administrator\\PycharmProjects\\untitled7\\picture\\' + file.name, 'wb') as f:  # 新建1张图片 ，图片名称为 上传的文件名
                for temp in file.chunks():  # 往图片添加图片信息
                    f.write(temp)
        model = load_model
        path = 'C:\\Users\\Administrator\\PycharmProjects\\untitled7\\picture\\' + file.name
        result = model.predict(path)
        cat = result[0][0]
        dog = result[0][1]
        if cat > dog:
            print("这是猫")
            dir = {"result": "猫"}
            return HttpResponse(json.dumps(dir))
        else:
            print("这是狗")
            dir = {"result": "狗"}
            return HttpResponse(json.dumps(dir))