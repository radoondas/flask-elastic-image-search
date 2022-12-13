from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class SearchForm(FlaskForm):
    searchbox = StringField('What are you looking for?', validators=[DataRequired()])
    submit = SubmitField('Search')
