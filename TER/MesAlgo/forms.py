from django import forms
class FigForm(forms.Form):
    perp = forms.FloatField(label='Perplexité',min_value=0)
    l_rate = forms.IntegerField(label='Learning Rate',min_value=1)
    nbiter=forms.IntegerField(label="Nombre d'itérations",min_value=250)
    csv=forms.ChoiceField(choices=[
        ('contact-lenses','contact-lenses'),
        ('pasture','pasture'),
        ('squash-stored','squash-stored'),
        ('newthyroid','newthyroid'),
        ('car','car'),
        ('bondrate','bondrate')]
  )