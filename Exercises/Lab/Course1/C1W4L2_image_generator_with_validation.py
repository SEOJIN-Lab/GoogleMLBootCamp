#아래 코드를 제외하고 C1W4L1 파일과 동일

#You need to download the data before running the code.
#!gdown --id 1onaG42NZft3wCE1WH0GDEbUhu75fedP5
#!gdown --id 1LYeusSEIiZQpwN-mthh5nKdA75VsKG1U


import os
import zipfile

local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./horse-or-human')

local_zip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./validation-horse-or-human')

zip_ref.close()