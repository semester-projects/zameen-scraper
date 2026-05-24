import os
import sys

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from scraper.spiders.zameen_spider import ZameenSpider


def execute_spider(limit: int = 10, raw_data_path: str = "assets/raw_data.csv", city: str = "Islamabad"):
    """
    Spawns and executes the Zameen spider crawling process.
    """
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "scraper.settings")
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    process.crawl(ZameenSpider, limit=limit, raw_data_path=raw_data_path, city=city)
    process.start()


if __name__ == "__main__":
    crawl_property_limit = 10
    output_raw_csv_path = "assets/raw_data.csv"
    city = "Islamabad"

    if len(sys.argv) > 1:
        crawl_property_limit = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_raw_csv_path = sys.argv[2]
    if len(sys.argv) > 3:
        city = sys.argv[3]

    execute_spider(limit=crawl_property_limit, raw_data_path=output_raw_csv_path, city=city)
