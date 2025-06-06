import pickle
from typing import List

from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def extract_listings_from_page(soup: BeautifulSoup):
    try:
        listings = soup.find_all(
            "a", attrs={"data-testid": lambda v: v and v.startswith("result-listing-")}
        )
        if not listings:
            print("No listings found on this page.")
            return []

        links = [listing["href"] for listing in listings]
        return ["https://suchen.mobile.de" + link for link in links]
    except Exception as e:
        print("Error extracting listings:", e)
        return []


def scrape_single_page(browser, url: str):
    """Scrapes a single page of listings."""
    page = browser.new_page()
    page.goto(url)
    page.wait_for_selector("body")

    # Try to accept cookies
    try:
        page.click("//button[contains(text(), 'Accept')]")
    except Exception as e:
        print("No cookie consent pop-up or unable to locate it:", e)

    # Small pause to let lazy-loaded content arrive
    page.wait_for_timeout(1_000)

    # Grab the full HTML
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    # Close the page after scraping
    page.close()
    return soup


def scrape_all_make_pages(make_id: str, max_pages: int = 50):
    """Scrapes all pages for a given make ID."""
    base_url = f"https://suchen.mobile.de/fahrzeuge/search.html?dam=false&isSearchRequest=true&p=%3A30000&s=Car&sb=rel&vc=Car&ms={make_id}&lang=en"

    all_listings = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        base_page = browser.new_page()
        for page_num in tqdm(range(1, max_pages + 1), desc=f"Scraping {make_id}"):
            url = f"{base_url}&pageNumber={page_num}"
            soup = scrape_single_page(browser, url)
            listings = extract_listings_from_page(soup)
            if listings and len(listings) > 0:
                all_listings.extend(listings)
            else:
                print(f"No listings found on page {page_num}")
                continue
        base_page.close()
        browser.close()
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


def scrape_all_listings_for_make(make_listings: List[str], make_name: str):
    """Scrapes details for all listings of a specific make."""
    all_details = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        base_page = browser.new_page()
        for listing_url in tqdm(make_listings, desc="Scraping listing details"):
            soup = scrape_single_page(browser, listing_url)
            try:
                # Check if the page is a valid listing
                if (
                    not soup.find("div", attrs={"data-testid": "vip-price-label"})
                    or not soup.find(
                        "article", attrs={"data-testid": "vip-technical-data-box"}
                    )
                    or not soup.find("ul", attrs={"data-testid": "vip-features-list"})
                ):
                    print(f"Skipping invalid listing: {listing_url}")
                    continue
                details = {
                    "make": make_name,
                    "price": extract_price_from_listing(soup),
                    "technical_details": extract_technical_details_from_listing(soup),
                    "additional_details": extract_additional_details_from_listing(soup),
                }
                all_details.append(details)
            except Exception as e:
                print(f"Error scraping listing {listing_url}: {e}")
                continue

        base_page.close()
        browser.close()

    return all_details


if __name__ == "__main__":
    makes = {
        "Audi": "1900",
        "Volkswagen": "25200",
        "Skoda": "22900",
        "Seat": "22500",
    }
    all_make_links = {}
    for make, make_id in makes.items():
        print(f"Scraping listings for {make} (ID: {make_id})")
        all_make_links[make] = []
        make_links = scrape_all_make_pages(make_id, max_pages=50)
        all_make_links[make].extend(make_links)
        print(f"Finished scraping listings for {make}.")

    with open("./data/playwright_make_links.pkl", "wb") as f:
        pickle.dump(all_make_links, f)

    print("All make links saved to ./data/playwright_make_links.pkl")

    # Now scrape details for each make
    all_make_listings = {}
    for make, links in all_make_links.items():
        print(f"Scraping details for {make} listings...")
        details = scrape_all_listings_for_make(links, make)
        all_make_listings[make] = details
        print(f"Finished scraping details for {make}.")

    with open("./data/playwright_listings.pkl", "wb") as f:
        pickle.dump(all_make_listings, f)
