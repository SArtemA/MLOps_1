import wget
from zipfile import ZipFile


wget.download('http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip')
zip = ZipFile('laboro_tomato.zip')
zip.extractall()
