import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
import sys


def get_url(url):
    r = requests.get(url)
    print(r)
    soup = BeautifulSoup(r.text, 'lxml')
    if (soup.find('div', class_ ='captcha__play-image') != None):
        print('captcha')
        sys.exit()
    return r.text


def get_data(html, data):
    soup = BeautifulSoup(html, 'lxml')
    ads = soup.find_all('div', class_ = 'n-snippet-cell2')
    for ad in ads:    
        try:
            brand = ad.find('div', class_ = 'n-snippet-cell2__brand-name').text.strip()
        except:
            brand = ''
        try:
            title = ad.find('div', class_ = 'n-snippet-cell2__title').find('a').get('title')
        except:
            title = ''
        try:
            price = ad.find('div', class_ = 'price').text.strip()
        except:
            price = ''
        id = ad.find('div', class_ = "n-snippet-cell2__more-prices-link").find('a').get('href').split('/')[2]
        data = data.append({'ID': id, 
                            'Brand': brand, 
                            'Title': title, 
                            'Price': price}, ignore_index = True)
#        get_description(id)
    return data


def get_description(id):
    name_data = []
    value_data = []
    
    desc_url = "https://market.yandex.ru/product/" + id + "/spec?track=tabs"
    name_data.append('ID')
    value_data.append(id)
    desc_html = get_url(desc_url)
    soup = BeautifulSoup(desc_html, 'lxml')
    ads = soup.find_all('div', class_ = "n-product-spec-wrap__body")
    
    for ad in ads:
        div_names = ad.find_all('dt', class_ = 'n-product-spec__name')
        for div in div_names:
            spec_name = div.find('span', class_ = 'n-product-spec__name-inner').text.strip().split('?')[0]
            name_data.append(spec_name)
        div_values = ad.find_all('dd', class_ = 'n-product-spec__value')
        for div in div_values:
            color_bool = 0
            color = ''
            if color_bool == 0:    
                color_values = div.find_all('div', class_ = 'product-color')
                for color_value in color_values:
                    color = color + '/' + color_value.get('title')
                if color != '':
                    color_bool = 1
                    value_data.append(color)
                    
            spec_value = div.find('span', class_ = 'n-product-spec__value-inner').text.strip()
            if (spec_value == '' or spec_value == ""):
                print("Empty")
            else: 
		value_data.append(spec_value)
      
    print(name_data)
#    if (color_bool == 1):
#        first = value_data[0]
#        value_data = [first] + value_data[1::2]
#    else: value_data = value_data[0::2] 
    
    print(value_data)    
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    spec_data = DataFrame({"Titles": name_data, "Values": value_data})
    write_excel(spec_data, id)
    
    
def write_excel(data, id):   
    data.to_excel("yandex_market_" + id + ".xlsx")
          
        
def main():
#    base_url = "https://market.yandex.ru/catalog--samokaty/54700/list?hid=7070735&glfilter=7081037%3A11854643&onstock=1&local-offers-first=0&"
#    page_part = "page="
    data = DataFrame([])
    
    for i in range(1, 5):
        url = base_url + page_part + str(i)
        html = get_url(url)
        data = get_data(html, data)
#        print(len(data))
	write_excel(data, i)

#    url = base_url + page_part + str(1)
#    html = get_url(url)    
#    data = get_data(html, data)        
#    f = open("ID.txt", 'r')
#    data = f.read().split('\n')
#    f.close()
#    for i in range(157, len(data)):    
#        get_description(data[i])
#    write_excel(data)


if __name__ == "__main__":
    main()