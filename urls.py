"""detection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.first),
    path('index',views.index),
    path('register/',views.register),
    path('register/registration',views.registration),
    path('login/',views.login),
    path('login/addlogin',views.addlogin),
    path('logout/',views.logout,name="logout"),
    path('v_users',views.v_users),
    path('test',views.test),
    #path('addfile',views.addfile),
    path('v_result',views.v_result),
    path('speech_sign',views.speech_sign,name="speech_sign"),
    #path('text_sign',views.text_sign,name="text_sign"),
    #path('web_cam',views.web_cam),
    path('livevoice',views.livevoice),




    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
