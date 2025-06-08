import os
import time
import pickle
import random
from typing import List, Dict

from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, BrowserContext
from playwright_stealth import stealth_sync


def make_stealth_context(pw, headless=False) -> BrowserContext:
    """
    Launches a browser and returns a stealth context.
    This version uses playwright-stealth for more robust evasion.
    """
    browser = pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--window-size=1920,1080",
        ],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"  # Updated User-Agent
        ),
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="Europe/Berlin",
    )
    # Apply stealth measures
    stealth_sync(context)
    return context


def extract_listings_from_page(soup: BeautifulSoup) -> List[str]:
    try:
        listings = soup.find_all(
            "a",
            attrs={"data-testid": lambda v: v and v.startswith("result-listing-")},
        )
        if not listings:
            return []
        links = [listing["href"] for listing in listings]
        # The hrefs are already absolute URLs, no need to prepend
        return links
    except Exception as e:
        print(f"Error extracting listings: {e}")
        return []


def scrape_single_page(page: Page, url: str) -> BeautifulSoup:
    """Scrapes a single page, reusing the same Page object."""
    print(f"Navigating to: {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    # More robust wait for the page to settle down
    page.wait_for_load_state("networkidle", timeout=30000)

    # Try to accept cookies if the button is visible
    cookie_button = page.locator("button:has-text('Accept')").first
    if cookie_button.is_visible():
        # print("Cookie consent button found. Clicking...")
        cookie_button.click()
        # Wait a moment for the overlay to disappear
        time.sleep(random.uniform(1, 2))

    # Add a small random delay to mimic human reading time
    time.sleep(random.uniform(1, 2))

    html = page.content()
    return BeautifulSoup(html, "html.parser")


def scrape_all_make_pages(
    context: BrowserContext, make_id: str, max_pages: int = 50
) -> List[str]:
    """Scrapes all pages for a given make ID."""
    base_url = f"https://suchen.mobile.de/fahrzeuge/search.html?dam=false&isSearchRequest=true&s=Car&sb=rel&vc=Car&ms={make_id}&lang=en"
    all_listings = []
    page = context.new_page()  # Use one page for the whole make

    for page_num in tqdm(range(1, max_pages + 1), desc=f"Scraping {make_id}"):
        url = f"{base_url}&pageNumber={page_num}"
        soup = scrape_single_page(page, url)
        listings = extract_listings_from_page(soup)
        listings_url = [f"https://suchen.mobile.de{link}" for link in listings]
        if listings_url:
            all_listings.extend(listings_url)
        else:
            print(f"No more listings found on page {page_num}. Stopping.")
            break  # Exit loop if a page has no listings

        # CRITICAL: Add a delay between scraping search result pages
        time.sleep(random.uniform(3.0, 6.0))

    page.close()
    return all_listings


def extract_price_from_listing(soup: BeautifulSoup):
    car_price_div = soup.find_all("div", attrs={"data-testid": "vip-price-label"})[0]
    car_price = car_price_div.find_all("div")[0].text
    return car_price


def extract_technical_details_from_listing(soup: BeautifulSoup):
    car_data = soup.find_all(
        "article", attrs={"data-testid": "vip-technical-data-box"}
    )[0]
    car_data = car_data.find_all("dl")[0]
    # extract all dt and dd tags
    car_data = car_data.find_all(["dt", "dd"])
    # zip them together
    car_data = list(zip(car_data[::2], car_data[1::2]))
    # make technical_data a dictionary
    technical_data = {dt.text: dd.text for dt, dd in car_data}
    return technical_data


def extract_additional_details_from_listing(soup: BeautifulSoup):
    features_list = soup.find_all("ul", attrs={"data-testid": "vip-features-list"})[0]
    features = [li.text for li in features_list.find_all("li")]
    return features


def scrape_all_listings_for_make(
    context: BrowserContext, make_listings: List[str], make_name: str
) -> List[Dict]:
    """Scrapes details for all listings of a specific make."""
    all_details = []
    page = context.new_page()  # Use one page for all listings of this make

    for listing_url in tqdm(make_listings, desc=f"Scraping details for {make_name}"):
        try:
            soup = scrape_single_page(page, listing_url)
            # Check if the page is a valid listing
            if not soup.find("div", attrs={"data-testid": "vip-price-label"}):
                print(f"Skipping invalid or blocked listing: {listing_url}")
                # Add a longer delay if we suspect we're being blocked
                time.sleep(random.uniform(10.0, 15.0))
                continue

            details = {
                "make": make_name,
                "price": extract_price_from_listing(soup),
                "technical_details": extract_technical_details_from_listing(soup),
                "additional_details": extract_additional_details_from_listing(soup),
            }
            all_details.append(details)

            # Save progress incrementally
            with open(f"./data/playwright_{make_name}_listings.pkl", "wb") as f:
                pickle.dump(all_details, f)

            # CRITICAL: Add a delay between scraping individual listings
            time.sleep(random.uniform(4.0, 8.0))

        except Exception as e:
            print(f"Error scraping listing {listing_url}: {e}")
            # Wait a bit longer after an error
            time.sleep(random.uniform(10.0, 20.0))
            continue

    page.close()
    return all_details


if __name__ == "__main__":
    makes = {
        "Audi": "1900",
        "Volkswagen": "25200",
        "Skoda": "22900",
        "Seat": "22500",
        "Ford": "9000",
        "Opel": "19000",
        "Toyota": "24100",
        "Mercedes-Benz": "17200",
    }
    # It's much more efficient to manage one Playwright instance
    with sync_playwright() as pw:
        context = make_stealth_context(pw, headless=False)

        # --- Part 1: Scrape all listing URLs ---
        all_make_links = {}
        if os.path.exists("./data/playwright_make_links.pkl"):
            print("Loading existing make links from file...")
            with open("./data/playwright_make_links.pkl", "rb") as f:
                all_make_links = pickle.load(f)
        else:
            for make, make_id in makes.items():
                print(f"Scraping listings for {make} (ID: {make_id})")
                make_links = scrape_all_make_pages(context, make_id, max_pages=50)
                all_make_links[make] = make_links
                print(f"Found {len(make_links)} listings for {make}.")

            with open("./data/playwright_make_links.pkl", "wb") as f:
                pickle.dump(all_make_links, f)
            print("All make links saved.")

        # --- Part 2: Scrape details from the URLs ---
        all_make_listings = {}
        for make, links in all_make_links.items():
            if not links:
                print(f"No links found for {make}, skipping detail scraping.")
                continue
            print(f"Scraping details for {make} listings...")
            details = scrape_all_listings_for_make(context, links, make)
            all_make_listings[make] = details
            print(f"Finished scraping details for {make}.")

        with open("./data/playwright_listings.pkl", "wb") as f:
            pickle.dump(all_make_listings, f)
        print("All listing details saved.")

        # Cleanly close the context and browser
        context.close()
        context.browser.close()
