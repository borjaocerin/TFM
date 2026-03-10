import { NavLink, Route, Routes } from "react-router-dom";
import { useEffect } from "react";

import { getActiveModel } from "./lib/api";
import { t } from "./lib/i18n";
import { useAppStore } from "./lib/state/appStore";
import { DatasetsPage } from "./routes/DatasetsPage";
import { TrainingPage } from "./routes/TrainingPage";
import { PredictPage } from "./routes/PredictPage";
import { OddsPage } from "./routes/OddsPage";

const navItems = [
  { to: "/", labelKey: "nav.datasets" },
  { to: "/training", labelKey: "nav.training" },
  { to: "/predict", labelKey: "nav.predict" },
  { to: "/odds", labelKey: "nav.odds" }
] as const;

function App() {
  const locale = useAppStore((state) => state.locale);
  const setLocale = useAppStore((state) => state.setLocale);
  const modelMetadata = useAppStore((state) => state.modelMetadata);
  const setModelMetadata = useAppStore((state) => state.setModelMetadata);
  const setToast = useAppStore((state) => state.setToast);
  const toast = useAppStore((state) => state.toast);

  useEffect(() => {
    getActiveModel()
      .then((response) => {
        setModelMetadata(response.model_available ? response.metadata ?? null : null);
      })
      .catch(() => {
        setModelMetadata(null);
      });
  }, [setModelMetadata]);

  return (
    <div className="app-shell">
      <div className="ambient-shape ambient-shape-left" />
      <div className="ambient-shape ambient-shape-right" />

      <header className="topbar">
        <div>
          <p className="kicker">TFM DEMO</p>
          <h1>LaLiga 1X2 Predictor</h1>
          <p className="subtitle">{t(locale, "header.subtitle")}</p>
        </div>

        <div className="topbar-right">
          <button
            className="locale-btn"
            onClick={() => setLocale(locale === "es" ? "en" : "es")}
            type="button"
          >
            {locale === "es" ? "EN" : "ES"}
          </button>
          <div className="status-card">
            <strong>{t(locale, "header.activeModel")}</strong>
            {modelMetadata ? (
              <>
                <span>{modelMetadata.best_model}</span>
                <span>{modelMetadata.trained_at}</span>
              </>
            ) : (
              <span>{t(locale, "header.noModel")}</span>
            )}
          </div>
        </div>
      </header>

      <nav className="nav-grid">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => (isActive ? "nav-link nav-link-active" : "nav-link")}
          >
            {t(locale, item.labelKey)}
          </NavLink>
        ))}
      </nav>

      <main className="content">
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/odds" element={<OddsPage />} />
        </Routes>
      </main>

      {toast && (
        <button className="toast" onClick={() => setToast(null)} type="button">
          {toast}
        </button>
      )}
    </div>
  );
}

export default App;
