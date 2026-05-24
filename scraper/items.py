import scrapy


class ZameenPropertyItem(scrapy.Item):
    price = scrapy.Field()
    area = scrapy.Field()
    city = scrapy.Field()
    bedrooms = scrapy.Field()
    bathrooms = scrapy.Field()
    location = scrapy.Field()
    property_type = scrapy.Field()
    built_in_year = scrapy.Field()
    parking_space = scrapy.Field()
    servant_quarters = scrapy.Field()
    store_rooms = scrapy.Field()
    kitchens = scrapy.Field()
    drawing_rooms = scrapy.Field()
