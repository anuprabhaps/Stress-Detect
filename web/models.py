from django.db import models

# Create your models here.

class Member(models.Model):
    username=models.CharField(max_length=30)
    password=models.CharField(max_length=12)
    city=models.CharField(max_length=30)
    mobile=models.IntegerField(max_length=10)

    def __str__(self):
        return self.firstname + " " + self.lastname