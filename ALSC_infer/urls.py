from django.urls import path

from . import views

urlpatterns = [
    path('query/', views.query, name='query'),
    path('showResult/', views.showResult, name='showResult')
]