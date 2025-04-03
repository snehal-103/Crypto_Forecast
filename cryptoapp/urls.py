from django.urls import path
from .views import user_list
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("news/", views.news, name="news"),
    path("predict/", views.predict, name="predict"),
    path("login/", views.user_login, name="login"),
    path("register/", views.user_register, name="register"),
    path("logout/", views.user_logout, name="logout"),
    path('users/', user_list, name='user_list'),
]
