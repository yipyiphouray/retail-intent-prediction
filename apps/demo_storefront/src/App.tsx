import { useCallback, useEffect, useMemo, useState } from "react";

import {
  type ClickPayload,
  type DemoApi,
  type Prediction,
  defaultDemoApi,
} from "./api";
import { CategoryCard } from "./components/CategoryCard";
import { ProductCard } from "./components/ProductCard";
import { PromoModal } from "./components/PromoModal";
import {
  PAGE_NUMBERS,
  catalogItems,
  categories,
  photographyLabels,
  type Category,
  type CatalogItem,
  locationLabels,
} from "./data/products";

const DEFAULT_COUNTRY = 29;
const CLICK_TRIGGER = 5;

type ViewMode = "shop" | "dashboard";
type RawRow = Record<string, number | string>;

const monthFromNow = () => {
  const month = new Date().getMonth() + 1;
  return 4 + (month % 5);
};

function toClickPayload(categoryCode: Category["code"], item: CatalogItem): ClickPayload {
  return {
    year: 2008,
    month: monthFromNow(),
    day: Math.min(Math.max(new Date().getDate(), 1), 31),
    country: DEFAULT_COUNTRY,
    page_1_main_category: categoryCode,
    page_2_clothing_model: item.modelCode,
    colour: item.colour,
    location: item.location,
    model_photography: item.modelPhotography,
    price: item.price,
    price_2: item.price >= 56 ? 1 : 2,
    page: item.page,
  };
}

interface AppProps {
  api?: DemoApi;
}

export function App({ api = defaultDemoApi }: AppProps): JSX.Element {
  const [viewMode, setViewMode] = useState<ViewMode>("shop");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [clickCount, setClickCount] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isStartingSession, setIsStartingSession] = useState(false);
  const [isSubmittingClick, setIsSubmittingClick] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
  const [activePage, setActivePage] = useState<(typeof PAGE_NUMBERS)[number]>(1);
  const [showAdModal, setShowAdModal] = useState(false);
  const [capturedRows, setCapturedRows] = useState<RawRow[]>([]);

  const startNewSession = useCallback(async () => {
    setIsStartingSession(true);
    setErrorMessage(null);

    try {
      const session = await api.createSession();
      setSessionId(session.session_id);
      setClickCount(session.click_count);
      setPrediction(session.prediction);
      setCapturedRows([]);
      setSelectedCategory(null);
      setActivePage(1);
      setShowAdModal(false);
      setViewMode("shop");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to start session.");
    } finally {
      setIsStartingSession(false);
    }
  }, [api]);

  useEffect(() => {
    void startNewSession();
  }, [startNewSession]);

  const visibleItems = useMemo(() => {
    if (activePage === 1 || selectedCategory === null) {
      return [];
    }

    return catalogItems
      .filter((item) => item.page === activePage && item.categoryCode === selectedCategory.code)
      .sort((a, b) => a.location - b.location);
  }, [activePage, selectedCategory]);

  const handlePageSelect = useCallback(
    (page: (typeof PAGE_NUMBERS)[number]) => {
      if (page === 1) {
        setActivePage(1);
        setErrorMessage(null);
        return;
      }

      if (selectedCategory === null) {
        setErrorMessage("Choose a category on page 1 before browsing product pages.");
        return;
      }

      setActivePage(page);
      setErrorMessage(null);
    },
    [selectedCategory],
  );

  const handleCategorySelect = useCallback((category: Category) => {
    setSelectedCategory(category);
    setActivePage(2);
    setErrorMessage(null);
  }, []);

  const handlePreviousCatalogPage = useCallback(() => {
    setActivePage((current) => {
      if (current <= 2) {
        return current;
      }
      return (current - 1) as 2 | 3 | 4 | 5;
    });
  }, []);

  const handleNextCatalogPage = useCallback(() => {
    setActivePage((current) => {
      if (current < 2 || current >= 5) {
        return current;
      }
      return (current + 1) as 2 | 3 | 4 | 5;
    });
  }, []);

  const handleProductClick = useCallback(
    async (item: CatalogItem) => {
      if (sessionId === null || selectedCategory === null) {
        return;
      }

      setIsSubmittingClick(true);
      setErrorMessage(null);

      try {
        const payload = toClickPayload(selectedCategory.code, item);
        const result = await api.postClick(sessionId, payload);

        setClickCount(result.click_count);
        setPrediction(result.prediction);
        setCapturedRows((rows) => [...rows, result.raw_row]);

        if (result.triggered && result.show_ad) {
          setShowAdModal(true);
        }
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : "Could not capture click.");
      } finally {
        setIsSubmittingClick(false);
      }
    },
    [api, selectedCategory, sessionId],
  );

  const predictionText = prediction
    ? `${prediction.label} (${Math.round(prediction.probability * 100)}%)`
    : "Not triggered yet";

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <PromoModal open={showAdModal} onClose={() => setShowAdModal(false)} />

      <main className="shop-shell" id="main-content">
        <header className="shop-header">
          <div>
            <p className="kicker">Retail Intent Demo</p>
            <h1>Bloom & Grace E-Shop</h1>
          </div>

          <div className="header-actions">
            <div className="view-switch" role="group" aria-label="View switcher">
              <button
                type="button"
                className={viewMode === "shop" ? "active" : ""}
                onClick={() => setViewMode("shop")}
              >
                Storefront
              </button>
              <button
                type="button"
                className={viewMode === "dashboard" ? "active" : ""}
                onClick={() => setViewMode("dashboard")}
              >
                Analyst Dashboard
              </button>
            </div>

            <button
              type="button"
              className="new-session-button"
              onClick={() => void startNewSession()}
              disabled={isStartingSession}
            >
              {isStartingSession ? "Starting…" : "Start New Session"}
            </button>
          </div>
        </header>

        {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}

        {viewMode === "shop" ? (
          <>
          <section className="page-nav-panel">
            <p className="panel-label">Navigation</p>
            <div className="page-tabs" role="tablist" aria-label="Store categories">
              {categories.map((category) => (
                <button
                  key={category.code}
                  type="button"
                  role="tab"
                  className={selectedCategory?.code === category.code ? "active" : ""}
                  aria-selected={selectedCategory?.code === category.code}
                  onClick={() => handleCategorySelect(category)}
                >
                  {category.name}
                </button>
              ))}
              <button
                type="button"
                role="tab"
                className={activePage === 1 ? "active" : ""}
                aria-selected={activePage === 1}
                onClick={() => handlePageSelect(1)}
              >
                Home
              </button>
            </div>

            <div className="selection-meta" aria-live="polite">
                <span>
                  Category: <strong>{selectedCategory?.name ?? "Not selected"}</strong>
                </span>
                <span>
                  Current page: <strong>{activePage}</strong>
                </span>
              </div>
            </section>

            {activePage === 1 ? (
              <section className="category-panel">
                <h2>Shop by Category</h2>
                <p>Select a category to continue browsing product photos.</p>
                <div className="category-grid">
                  {categories.map((category) => (
                    <CategoryCard
                      key={category.code}
                      category={category}
                      onChoose={handleCategorySelect}
                    />
                  ))}
                </div>
              </section>
            ) : (
              <section className="catalog-panel">
                <h2>{selectedCategory ? `${selectedCategory.name} Gallery` : "Product Gallery"}</h2>
                <p>
                  Each photo click emits one API row with product code, colour code, location,
                  model photography orientation, price, and page context.
                </p>
                <div className="catalog-controls">
                  <button
                    type="button"
                    onClick={handlePreviousCatalogPage}
                    disabled={activePage <= 2}
                  >
                    Previous
                  </button>
                  <span>Catalog page {activePage} of 5</span>
                  <button
                    type="button"
                    onClick={handleNextCatalogPage}
                    disabled={activePage >= 5}
                  >
                    Next
                  </button>
                </div>

                {selectedCategory === null ? (
                  <div className="empty-state">Return to page 1 and choose a category first.</div>
                ) : (
                  <div className="photo-grid">
                    {visibleItems.map((item, index) => (
                      <ProductCard
                        key={item.id}
                        item={item}
                        index={index}
                        disabled={isSubmittingClick || isStartingSession}
                        onSelect={(chosen) => void handleProductClick(chosen)}
                      />
                    ))}
                  </div>
                )}
              </section>
            )}
          </>
        ) : (
          <section className="dashboard-panel">
            <h2>Session Behavior Dashboard</h2>
            <p>
              Analytics and captured clickstream rows are separated from the shopper-facing UI.
            </p>

            <div className="dashboard-metrics">
              <article>
                <span>Session ID</span>
                <strong>{sessionId ?? "-"}</strong>
              </article>
              <article>
                <span>Clicks Captured</span>
                <strong>{clickCount}</strong>
              </article>
              <article>
                <span>Prediction</span>
                <strong>{predictionText}</strong>
              </article>
              <article>
                <span>Trigger Rule</span>
                <strong>{CLICK_TRIGGER} clicks (once)</strong>
              </article>
            </div>

            {capturedRows.length === 0 ? (
              <div className="empty-state">No captured rows yet. Browse the storefront first.</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Order</th>
                      <th>Page</th>
                      <th>Category</th>
                      <th>Model</th>
                      <th>Location</th>
                      <th>Pose</th>
                      <th>Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {capturedRows.map((row) => {
                      const locationCode = Number(row["location"]);
                      const poseCode = Number(row["model photography"]);
                      return (
                        <tr key={`${row["session ID"]}-${row["order"]}`}>
                          <td>{row["order"]}</td>
                          <td>{row["page"]}</td>
                          <td>{row["page 1 (main category)"]}</td>
                          <td>{row["page 2 (clothing model)"]}</td>
                          <td>{locationLabels[locationCode as keyof typeof locationLabels]}</td>
                          <td>{photographyLabels[poseCode as keyof typeof photographyLabels]}</td>
                          <td>${Number(row["price"]).toFixed(0)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </main>
    </>
  );
}
