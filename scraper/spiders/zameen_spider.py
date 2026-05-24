import json
import re
import scrapy
from typing import Generator, Any, Dict
from scraper.items import ZameenPropertyItem
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn
)


class ZameenSpider(scrapy.Spider):
    """
    A highly optimized Scrapy spider targeting Zameen.com property listings.
    Uses native Scrapy selector engines (CSS/XPath) and integrates a premium
    Rich terminal UI to show scraping progress in real time.
    """
    name = "zameen"

    def __init__(
        self,
        limit: int = 10,
        raw_data_path: str = "assets/raw_data.csv",
        city: str = "Islamabad",
        *args: Any,
        **kwargs: Any
    ) -> None:
        super(ZameenSpider, self).__init__(*args, **kwargs)
        self.crawl_limit = int(limit)
        self.raw_data_path = raw_data_path
        self.city = city

        # Track items for limit handling and visual progress
        self.total_properties_scraped = 0
        self.total_properties_enqueued = 0
        self.page_number = 1

        # Load configuration start URLs
        with open("scraper/config.json", "r") as config_file:
            self.config = json.load(config_file)

        city_urls = self.config.get("start_urls", {})
        self.start_url = city_urls.get(
            self.city,
            city_urls.get(
                "Islamabad",
                "https://www.zameen.com/Homes/Islamabad-3-1.html"
            )
        )
        self.start_urls = [self.start_url]

        # Initialize premium Rich Terminal UI progress visualizer
        self.progress = Progress(
            TextColumn("[bold magenta]✨ {task.description}[/bold magenta]"),
            BarColumn(
                bar_width=40,
                style="bright_black",
                complete_style="green",
                finished_style="cyan"
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(
                "⚡ [bold yellow]{task.completed}/{task.total} "
                "items[/bold yellow]"
            ),
            TimeRemainingColumn()
        )
        self.progress.start()
        total_limit = self.crawl_limit if self.crawl_limit > 0 else 1000
        self.task = self.progress.add_task(
            f"Crawling {self.city} Listings",
            total=total_limit
        )

    def closed(self, reason: str) -> None:
        """Stops the Rich progress visualization when the spider terminates."""
        self.progress.stop()

    def start_requests(self) -> Generator[scrapy.Request, None, None]:
        """Triggers the initial scraping crawl requests."""
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                errback=self.handle_error
            )

    def handle_error(self, failure: Any) -> None:
        """Logs standard request failures."""
        self.logger.error(f"HTTP request failed: {str(failure)}")

    def parse(
        self,
        response: scrapy.http.Response
    ) -> Generator[scrapy.Request, None, None]:
        """
        Parses Zameen.com search results listing page.
        Extracts high-level details and schedules detail page crawls.
        """
        # Guard against Cloudflare or access blocks
        if (
            response.status in {403, 503, 429}
            or "cloudflare" in response.text.lower()
        ):
            self.logger.error(
                f"Access blocked by Zameen.com security gates "
                f"(Status: {response.status})"
            )
            return

        # Extract listings using native high-performance Scrapy CSS Selectors
        extracted_listing_cards = response.css('[aria-label="Listing"]')

        if not extracted_listing_cards:
            self.progress.console.print(
                "[bold yellow]⚠ No more listing cards found on this page."
                "[/bold yellow]"
            )
            return

        for listing_card_element in extracted_listing_cards:
            # Enforce limits if set
            if 0 < self.crawl_limit <= self.total_properties_enqueued:
                break

            scraped_property_item = ZameenPropertyItem()
            scraped_property_item['city'] = self.city

            # Extract basic text fields using Scrapy CSS selectors
            raw_title = (
                listing_card_element
                .css('[aria-label="Title"]::text')
                .get(default="")
                .strip()
                .lower()
            )
            scraped_property_item['price'] = (
                listing_card_element
                .css('[aria-label="Price"]::text')
                .get(default="")
                .strip()
            )
            scraped_property_item['location'] = (
                listing_card_element
                .css('[aria-label="Location"]::text')
                .get(default="")
                .strip()
            )
            scraped_property_item['bedrooms'] = (
                listing_card_element
                .css('[aria-label="Beds"]::text')
                .get(default="")
                .strip()
            )
            scraped_property_item['bathrooms'] = (
                listing_card_element
                .css('[aria-label="Baths"]::text')
                .get(default="")
                .strip()
            )
            scraped_property_item['area'] = (
                listing_card_element
                .css('[aria-label="Area"]::text')
                .get(default="")
                .strip()
            )

            # Heuristically classify the property type from the card title
            if "flat" in raw_title or "apartment" in raw_title:
                scraped_property_item['property_type'] = "Flat"
            elif "portion" in raw_title:
                scraped_property_item['property_type'] = "Portion"
            elif "penthouse" in raw_title:
                scraped_property_item['property_type'] = "Penthouse"
            else:
                scraped_property_item['property_type'] = "House"

            # Extract detail page URL link
            property_detail_url = (
                listing_card_element.css('a::attr(href)').get()
            )
            if property_detail_url:
                self.total_properties_enqueued += 1
                detail_url = response.urljoin(property_detail_url)
                yield scrapy.Request(
                    detail_url,
                    callback=self.parse_detail,
                    meta={'item': scraped_property_item},
                    errback=self.handle_detail_error
                )
            else:
                # Default empty values if link is absent
                fields = (
                    'built_in_year', 'parking_space', 'servant_quarters',
                    'store_rooms', 'kitchens', 'drawing_rooms'
                )
                for field in fields:
                    scraped_property_item[field] = ""

                self.total_properties_scraped += 1
                self.total_properties_enqueued += 1
                self.progress.update(self.task, advance=1)

                self.progress.console.print(
                    f"[bold green]✔[/bold green] Scraped "
                    f"[cyan]{scraped_property_item['property_type']}[/cyan] "
                    f"(No details) in "
                    f"[magenta]{scraped_property_item['location']}[/magenta]"
                )
                yield scraped_property_item

        # Pagination: Build the next search page URL and continue crawling
        max_pages = (
            (self.crawl_limit // 25) + 1 if self.crawl_limit > 0 else 100
        )
        if (
            self.page_number < max_pages
            and (
                self.crawl_limit <= 0
                or self.total_properties_enqueued < self.crawl_limit
            )
        ):
            self.page_number += 1
            url_parts = self.start_url.split("-")
            if len(url_parts) >= 3:
                next_page_url = (
                    "-".join(url_parts[:-1]) + f"-{self.page_number}.html"
                )
                yield scrapy.Request(
                    next_page_url,
                    callback=self.parse,
                    errback=self.handle_error
                )

    def parse_detail(
        self,
        response: scrapy.http.Response
    ) -> Generator[ZameenPropertyItem, None, None]:
        """
        Parses the specific property details page.
        Extracts deep features from Zameen's JSON application state.
        """
        scraped_property_item = response.meta['item']

        # Initialize default empty values for detail attributes
        fields = (
            'built_in_year', 'parking_space', 'servant_quarters',
            'store_rooms', 'kitchens', 'drawing_rooms'
        )
        for field in fields:
            scraped_property_item[field] = ""

        # Handle security blocking or missing body response
        if (
            response.status in {403, 503, 429}
            or "cloudflare" in response.text.lower()
        ):
            self.total_properties_scraped += 1
            self.progress.update(self.task, advance=1)
            yield scraped_property_item
            return

        try:
            # Query Zameen's window.state JSON configuration in scripts
            javascript_state_content = response.xpath(
                '//script[contains(text(), "window.state =")]/text()'
            ).get()

            if javascript_state_content:
                # RegEx search for the Javascript state assignment object
                regex_state_match = re.search(
                    r'window\.state\s*=\s*(\{.*?\});\s*window\.',
                    javascript_state_content,
                    re.DOTALL
                )
                if not regex_state_match:
                    regex_state_match = re.search(
                        r'window\.state\s*=\s*(\{.*?\});',
                        javascript_state_content,
                        re.DOTALL
                    )

                if regex_state_match:
                    deserialized_state_data = json.loads(
                        regex_state_match.group(1)
                    )
                    property_detail_data = (
                        deserialized_state_data
                        .get("property", {})
                        .get("data", {})
                    )
                    amenities_category_groups = (
                        property_detail_data.get("amenities", [])
                    )

                    # Parse high-fidelity amenities
                    extracted_amenities_mapping: Dict[str, str] = {}
                    for group in amenities_category_groups:
                        for amenity in group.get('amenities', []):
                            slug = amenity.get('slug', '').lower()
                            value = amenity.get('value', '')
                            # Empty value stands for boolean True or count of 1
                            if value == "":
                                value = "1"
                            extracted_amenities_mapping[slug] = value

                    # Map extracted values to items
                    if 'built-in-year' in extracted_amenities_mapping:
                        scraped_property_item['built_in_year'] = (
                            extracted_amenities_mapping['built-in-year']
                        )
                    if 'parking-spaces' in extracted_amenities_mapping:
                        scraped_property_item['parking_space'] = (
                            extracted_amenities_mapping['parking-spaces']
                        )
                    if 'servant-quarters' in extracted_amenities_mapping:
                        scraped_property_item['servant_quarters'] = (
                            extracted_amenities_mapping['servant-quarters']
                        )
                    if 'store-rooms' in extracted_amenities_mapping:
                        scraped_property_item['store_rooms'] = (
                            extracted_amenities_mapping['store-rooms']
                        )
                    if 'kitchens' in extracted_amenities_mapping:
                        scraped_property_item['kitchens'] = (
                            extracted_amenities_mapping['kitchens']
                        )
                    if 'drawing-room' in extracted_amenities_mapping:
                        scraped_property_item['drawing_rooms'] = (
                            extracted_amenities_mapping['drawing-room']
                        )

                    # Override count if detailed counts are present
                    if (
                        'bedrooms' in extracted_amenities_mapping
                        and extracted_amenities_mapping['bedrooms']
                    ):
                        scraped_property_item['bedrooms'] = (
                            extracted_amenities_mapping['bedrooms']
                        )
                    if (
                        'bathrooms' in extracted_amenities_mapping
                        and extracted_amenities_mapping['bathrooms']
                    ):
                        scraped_property_item['bathrooms'] = (
                            extracted_amenities_mapping['bathrooms']
                        )

        except Exception as error_msg:
            self.logger.error(
                f"Detail page JSON parser exception: {str(error_msg)}"
            )

        self.total_properties_scraped += 1
        self.progress.update(self.task, advance=1)

        # Log a beautiful premium micro-card feed to the console
        scraped_price_value = scraped_property_item.get('price', 'N/A')
        scraped_location_value = scraped_property_item.get('location', 'N/A')
        self.progress.console.print(
            f"[bold green]✔[/bold green] Scraped "
            f"[cyan]{scraped_property_item['property_type']}[/cyan] | "
            f"[magenta]{scraped_location_value}[/magenta] | "
            f"[yellow]{scraped_price_value}[/yellow]"
        )

        yield scraped_property_item

    def handle_detail_error(
        self,
        failure: Any
    ) -> Generator[ZameenPropertyItem, None, None]:
        """
        Callback handles detail page connection errors,
        yielding base listing item.
        """
        scraped_property_item = failure.request.meta['item']

        fields = (
            'built_in_year', 'parking_space', 'servant_quarters',
            'store_rooms', 'kitchens', 'drawing_rooms'
        )
        for field in fields:
            scraped_property_item[field] = ""

        self.total_properties_scraped += 1
        self.progress.update(self.task, advance=1)
        yield scraped_property_item
