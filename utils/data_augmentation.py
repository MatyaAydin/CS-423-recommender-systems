import isbnlib #pip install isbnlib
import pandas as pd
import pickle


df_book = pd.read_csv('./data/books.csv')
ISBNs = df_book['ISBN'].values

metadata = []
prb_isbn = []

###Extracting informations###

#warning: this takes a while
for i,isbn in enumerate(ISBNs):
    isbn_str = str(isbn)
    try: 
        book_info = isbnlib.meta(isbn_str)
        metadata.append(book_info)

    #either empty string or non existing ID
    except:
        prb_isbn.append(isbn_str)
    if i % 1000 == 0:
        print(f'done with {i}')



with open("books_metadata", "wb") as fp:
    pickle.dump(metadata, fp)


with open("prb_isbn", "wb") as fp:
    pickle.dump(prb_isbn, fp)


###Loading and augmenting###


with open("books_metadata", "rb") as fp:
  metadata = pickle.load(fp)


with open("prb_isbn", "rb") as fp:
  prb = pickle.load(fp)



df_filtered = df_book[~df_book['ISBN'].astype(str).isin(prb)]



isbns = df_filtered['ISBN'].values
titles = []
authors = []
publishers = []
years = []
languages = []

del_isbn = []

for i in range(len(isbns)):
  #we still have no info for some books that isbnlib recognized
  if len(metadata[i]) == 0:
    del_isbn.append(isbns[i])
  else:
    titles.append(metadata[i]['Title'])
    authors.append(metadata[i]['Authors'])
    publishers.append(metadata[i]['Publisher'])
    years.append(metadata[i]['Year'])
    languages.append(metadata[i]['Language'])

#only get books that we have informations on
df_augmented = df_filtered[~df_filtered['ISBN'].isin(del_isbn)].copy()
#adds features
df_augmented['title'] = titles
df_augmented['author'] = authors
df_augmented['publisher'] = publishers
df_augmented['year'] = years
df_augmented['language'] = languages

#save
df_augmented.to_csv('../data/books_augmented.csv')



