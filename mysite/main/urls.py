from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("Rlogin", views.r_login, name="r_login"),
    path("interview", views.interviewDirect, name="interview"),
    # path("interview", views.interview, name="interview"),
    path("success", views.success, name="success"),
    path("candidates", views.candidates_sub, name="candidates_sub"),
    path('candidate/<int:candidate_id>/result', views.view_candidate_result, name='view_result'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
