# Generated by Django 3.2.8 on 2021-10-08 14:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MesAlgo', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='carousel',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='slide/'),
        ),
    ]
