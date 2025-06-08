import os
import pickle
import random
import time
from typing import List, Dict

from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, BrowserContext
from playwright_stealth import stealth_sync


def make_stealth_context(pw, headless=False) -> BrowserContext:
    """
    Launches a heavily stealthed Chromium browser.
    """
    browser = pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",  # Use maximized window
        ],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        ## FIX: Use a maximized viewport and a German locale to match the target site
        viewport=None,  # None for maximized window
        locale="de-DE",
        timezone_id="Europe/Berlin",
    )
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
        return links
    except Exception as e:
        print(f"Error extracting listings: {e}")
        return []


def scrape_single_page(page: Page, url: str) -> BeautifulSoup:
    """Scrapes a single page with more human-like interactions."""
    page.goto(url, wait_until="networkidle", timeout=90000)

    cookie_button_selector = "button:has-text('Accept')"
    try:
        # Wait for the button to be potentially visible, but don't fail if not
        page.wait_for_selector(cookie_button_selector, timeout=5000)
        cookie_button = page.locator(cookie_button_selector).first
        if cookie_button.is_visible():
            # Move mouse over the button first
            cookie_button.hover()
            time.sleep(random.uniform(0.2, 0.3))
            cookie_button.click()
            # Wait for the overlay to disappear
            time.sleep(random.uniform(0.3, 0.5))
    except Exception:
        print("Cookie button not found or timed out, continuing.")
        pass

    # Simulate reading time
    time.sleep(random.uniform(0.2, 0.5))

    html = page.content()
    return BeautifulSoup(html, "html.parser")


def scrape_all_make_pages(
    context: BrowserContext, make_id: str, max_pages: int = 50
) -> List[str]:
    """Scrapes all pages for a given make ID with slower, safer pacing."""
    base_url = f"https://suchen.mobile.de/fahrzeuge/search.html?dam=false&isSearchRequest=true&s=Car&sb=rel&vc=Car&ms={make_id}&lang=en"
    all_listings = []
    page = context.new_page()

    for page_num in tqdm(range(1, max_pages + 1), desc=f"Scraping {make_id}"):
        url = f"{base_url}&pageNumber={page_num}"
        soup = scrape_single_page(page, url)
        relative_links = extract_listings_from_page(soup)

        if relative_links:
            absolute_links = [
                f"https://suchen.mobile.de{link}" for link in relative_links
            ]
            all_listings.extend(absolute_links)
        else:
            print(f"No more listings found on page {page_num}. Stopping.")
            break

        # FIX: Increased delay between scraping search result pages
        print("Taking a longer break between search pages...")
        time.sleep(random.uniform(8.0, 15.0))

    page.close()
    return all_listings


# --- The data extraction functions remain the same ---
def extract_price_from_listing(soup: BeautifulSoup):
    try:
        car_price_div = soup.find("div", attrs={"data-testid": "vip-price-label"})
        car_price = car_price_div.find("div").text
        return car_price
    except (AttributeError, IndexError): return "Price not found"

def extract_technical_details_from_listing(soup: BeautifulSoup):
    try:
        car_data_article = soup.find("article", attrs={"data-testid": "vip-technical-data-box"})
        car_data_dl = car_data_article.find("dl")
        car_data_items = car_data_dl.find_all(["dt", "dd"])
        car_data_pairs = list(zip(car_data_items[::2], car_data_items[1::2]))
        return {dt.text.strip(): dd.text.strip() for dt, dd in car_data_pairs}
    except (AttributeError, IndexError): return {}

def extract_additional_details_from_listing(soup: BeautifulSoup):
    try:
        features_list = soup.find("ul", attrs={"data-testid": "vip-features-list"})
        return [li.text.strip() for li in features_list.find_all("li")]
    except (AttributeError, IndexError): return []


def scrape_all_listings_for_make(
    pw, make_listings: List[str], make_name: str
) -> List[Dict]:
    all_details = []
    for listing_url in tqdm(make_listings, desc=f"Scraping details for {make_name}"):
        try:
            # Create a new browser context for each listing
            context = make_stealth_context(pw, headless=False)
            page = context.new_page()
            
            soup = scrape_single_page(page, listing_url)
            if not soup.find("div", attrs={"data-testid": "vip-price-label"}):
                print(f"Skipping invalid or blocked listing: {listing_url}")
                time.sleep(random.uniform(15.0, 25.0))
                context.close()
                context.browser.close()
                continue
                
            details = {
                "make": make_name,
                "price": extract_price_from_listing(soup),
                "technical_details": extract_technical_details_from_listing(soup),
                "additional_details": extract_additional_details_from_listing(soup),
            }
            all_details.append(details)
            os.makedirs("./data", exist_ok=True)
            with open(f"./data/playwright_{make_name}_details.pkl", "wb") as f:
                pickle.dump(all_details, f)
                
            # Close the context and browser for this listing
            page.close()
            context.close()
            context.browser.close()
            
            # FIX: Increased delay between scraping individual listings
            time.sleep(random.uniform(1.0, 2.0))
        except Exception as e:
            print(f"Error scraping listing {listing_url}: {e}")
            time.sleep(random.uniform(1.0, 2.0))
            # Make sure to close the context and browser even if there's an error
            try:
                page.close()
                context.close()
                context.browser.close()
            except:
                pass
            continue
    return all_details


if __name__ == "__main__":
    makes = {
        "Audi": "1900",
        "Volkswagen": "25200",
        "Skoda": "22900",
        "Seat": "22500",
    }
    with sync_playwright() as pw:
        # --- Part 1: Scrape listing URLs ---
        all_make_links = {}
        all_make_links: Dict[str, List[str]] = {}
        if os.path.exists("./data/new_short_make_links.pkl"):
            with open("./data/new_short_make_links.pkl", "rb") as f:
                all_make_links = pickle.load(f)
        else:
            context = make_stealth_context(pw, headless=False)
            for make, make_id in makes.items():
                make_links = scrape_all_make_pages(context, make_id, max_pages=50)
                all_make_links[make] = make_links
            with open("./data/new_short_make_links.pkl", "wb") as f:
                pickle.dump(all_make_links, f)
            context.close()
            context.browser.close()

        # --- Part 2: Scrape details ---
        for make, links in all_make_links.items():
            if links:
                scrape_all_listings_for_make(pw, links, make)