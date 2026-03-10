import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { useEffect } from "react";

import { getActiveModel } from "./lib/api";
import { t } from "./lib/i18n";
import { useAppStore } from "./lib/state/appStore";
import { DatasetsPage } from "./routes/DatasetsPage";
import { OddsPage } from "./routes/OddsPage";

const navItems = [
  { to: "/", labelKey: "nav.home" },
  { to: "/odds", labelKey: "nav.odds" }
] as const;

function App() {
  const locale = useAppStore((state) => state.locale);
  const setLocale = useAppStore((state) => state.setLocale);
  const modelMetadata = useAppStore((state) => state.modelMetadata);
  const setModelMetadata = useAppStore((state) => state.setModelMetadata);
  const setToast = useAppStore((state) => state.setToast);
  const toast = useAppStore((state) => state.toast);

  const bestModel =
    modelMetadata && typeof modelMetadata["best_model"] === "string"
      ? modelMetadata["best_model"]
      : "-";
  const trainedAt =
    modelMetadata && typeof modelMetadata["trained_at"] === "string"
      ? modelMetadata["trained_at"]
      : "-";

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
                <span>{bestModel}</span>
                <span>{trainedAt}</span>
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
          <Route path="/odds" element={<OddsPage />} />
          <Route path="/training" element={<Navigate to="/" replace />} />
          <Route path="/predict" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
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
