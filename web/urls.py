from django.urls import include, re_path as url
from . import views

urlpatterns = [
    url(r'^$', views.register, name='register'),
    url(r'^login/$', views.login, name='login'),
    url(r'^details/$', views.details, name='details'),
    url(r'^predict/$', views.predict, name='predict'),
]