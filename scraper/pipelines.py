import csv
import os


class ZameenCSVPipeline:
    """
    Scrapy item pipeline that serializes collected property listings
    into standard formatted CSV structures.
    """

    def open_spider(self, spider):
        self.file_path = getattr(spider, "raw_data_path", "assets/raw_data.csv")
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        self.file = open(self.file_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "Price",
                "Area",
                "City",
                "Bedrooms",
                "Bathrooms",
                "Location",
                "Property type",
                "Built in year",
                "Parking space",
                "Servant Quarters",
                "Store rooms",
                "Kitchens",
                "Drawing Rooms",
            ]
        )

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(
            [
                item.get("price", ""),
                item.get("area", ""),
                item.get("city", "Islamabad"),
                item.get("bedrooms", ""),
                item.get("bathrooms", ""),
                item.get("location", ""),
                item.get("property_type", ""),
                item.get("built_in_year", ""),
                item.get("parking_space", ""),
                item.get("servant_quarters", ""),
                item.get("store_rooms", ""),
                item.get("kitchens", ""),
                item.get("drawing_rooms", ""),
            ]
        )
        return item
