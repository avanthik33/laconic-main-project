from django.db import models


class Paragraph(models.Model):
    content = models.TextField()

    def __str__(self):
        return self.content[:50]  # Display first 50 characters of content for admin


class Question(models.Model):
    paragraph = models.ForeignKey(Paragraph, on_delete=models.CASCADE)
    text = models.TextField()


class Option(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    text = models.CharField(max_length=255)
    is_correct = models.BooleanField(default=False)


