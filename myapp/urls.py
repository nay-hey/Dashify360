from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('registration/', views.registration, name='registration'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('dashboard/', views.dashboard, name='dashboard'), 
    path('index/', views.index, name="index"),
    path('ask_question/', views.ask_question, name="ask_question"),
    path('dashboard_chartjs/', views.dashboard_chartjs, name="dashboard_chartjs"),
    path('index_chartjs/', views.index_chartjs, name="index_chartjs"),
    path('ml_output/', views.ml_output, name="ml_output"),
    path('download_csv/', views.download_csv, name='download_csv'),
    
]
