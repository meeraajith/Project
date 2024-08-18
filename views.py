from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
from django.shortcuts import redirect
from django.urls import reverse
from  django.core.files.storage import FileSystemStorage
import datetime

from .models import *


from ML import main2,record_audio
#from ML.classify_webcam import predict
from keras import backend as K
from ML import test1

import os

def first(request):
    return render(request,'index.html')

def index(request):
    return render(request,'index.html')

def register(request):
    return render(request,'register.html')

def registration(request):
    if request.method=="POST":
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')

        reg=registerr(name=name,email=email,password=password)
        reg.save()
    return render(request,'index.html')

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == 'admin@gmail.com' and password =='admin':
        request.session['logintdetail'] = email
        request.session['admin'] = 'admin'
        return render(request,'index.html')
    elif registerr.objects.filter(email=email,password=password).exists():
        userdetails=registerr.objects.get(email=request.POST['email'], password=password)
        if userdetails.password == request.POST['password']:
            request.session['uid'] = userdetails.id
        return render(request,'index.html')
    else:
        return render(request, 'login.html', {'success':'Invalid email id or Password'})
    
def logout(request):
    session_keys = list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(first)

def v_users(request):
    user = registerr.objects.all()
    return render(request, 'viewusers.html', {'result': user})

def test(request):
    return render(request,'test.html')

def speech_sign(request):
    main2.start_listening()
    return render(request,'test.html')

'''def text_sign(request):
    if request.method=="POST":
        text_data = request.POST.get('text_data')
        #main3.start_listening(base_dir="ML/", input_text=text_data)
        return render(request,'test.html')
    else:
        return render(request,'test.html')'''

'''def text_sign(request):
    if request.method=="POST":
        u_id = request.session['uid']
        myfile = request.FILES['image']
        try:
            os.remove("media/input/test.jpg")
        except:
            pass
        fs = FileSystemStorage(location="media/input")
        fs.save("test.jpg",myfile)
        fs = FileSystemStorage()
        fs.save(myfile.name,myfile)
        record_audio.record()
        result=classify.classify_image()
        print(result)
        cus=uploads(u_id=u_id,result=result,file=myfile.name)
        #cus=uploads(u_id=u_id,result=result,file="test.wav")
        cus.save()

        
    return render(request,'test.html',{'result':result})'''
    

def web_cam(request):
    if request.method == "POST":
        result =predict()
        return render(request, 'web_cam.html')
    else:
        return render(request, 'web_cam.html')


    
    
def v_result(request):
    uid=request.session['uid']
    user = uploads.objects.filter(u_id=uid)
    return render(request, 'viewresult.html', {'result': user})

'''def livevoice(request):
   # stud=request.session['sname']
    #K.clear_session()
    emotion=test1.predict()
    K.clear_session()
    print("the predicted emotion is:",emotion)
    user=result_tbl.objects.filter(emotion=emotion)
    ins=result_tbl(user=stud,uid= request.session['uid'] ,emotion=emotion,mode='Voice')    
    ins.save()
    return render(request,'viewresults.html',{'res':user,'emtn':emotion})'''

def livevoice(request): 
        
        emotion = test1.predict()
        
        print("Predicted emotion:", emotion)
       
        #user = request.objects.filter(emotion=emotion)
        #print("Filtered user queryset:", user)

        #ins = result_tbl( uid=str(request.session['uid']), emotion=emotion, mode='Voice')
        #ins.save()

        return render(request, 'test.html', { 'emtn': emotion})  
   




