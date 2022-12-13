from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField


class InputFileForm(FlaskForm):
    file = FileField()
    submit = SubmitField('Submit')
