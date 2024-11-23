
import pandas as pd
import requests


df_book = pd.read_csv('./data/books.csv')

df_book['ISBN'] = df_book['ISBN'].astype(str)
#adds missing zeros
df_book['ISBN'] = df_book['ISBN'].apply(lambda x: '0'*(10 - len(x))+ x)

ISBNs = df_book['ISBN'].values
book_ids = df_book['book_id']




titles = []
summaries = []
genres = []
authors = []
publishers = []
languages = []
dates = []

valid_isbn = []
unknown_isbn = []
valid_book_id = []

#warning: this takes a while
for i, isbn in enumerate(ISBNs):
    url = f'https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}'
    try:
        response = requests.get(url)
        book_data = response.json()

        if "items" in book_data:
            book_info = book_data["items"][0]["volumeInfo"]

            summary = book_info.get("description", "Summary not available")
            genre = book_info.get("categories", "Genre not available")
            title = book_info.get("title", "Title not available")
            publisher = book_info.get("publisher", "Publisher not available")
            language = book_info.get("language", "Language not available")
            published_date = book_info.get("publishedDate", "Year not available")

            summaries.append(summary)
            genres.append(genre)
            titles.append(title)
            publishers.append(publisher)
            languages.append(language)
            dates.append(published_date)

            valid_isbn.append(isbn)
            valid_book_id.append(book_ids[i])


        else:
            unknown_isbn.append(isbn)

    except:
            unknown_isbn.append(isbn)
    
    if i % 1000 == 0:
        print(f'done with {i}')
        print(len(summaries), len(unknown_isbn), '\n')


df_request = pd.DataFrame({"ISBN": valid_isbn, 'book_id': valid_book_id, 'title': titles, 'genre': genres, 'summary': summaries, 'publisher': publishers, 'language': languages, 'date': dates})

df_request.to_csv('../data/books_requests.csv', index=False)