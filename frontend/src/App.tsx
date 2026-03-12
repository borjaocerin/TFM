import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { t } from "./lib/i18n";
import { useAppStore } from "./lib/state/appStore";
import { DatasetsPage } from "./routes/DatasetsPage";
import { MatchPredictionPage } from "./routes/MatchPredictionPage";

const navItems = [
  { to: "/", labelKey: "nav.home" }
] as const;

function App() {
  const locale = useAppStore((state) => state.locale);
  const setLocale = useAppStore((state) => state.setLocale);
  const setToast = useAppStore((state) => state.setToast);
  const toast = useAppStore((state) => state.toast);

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
        </div>
      </header>

      {navItems.length > 1 && (
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
      )}

      <main className="content">
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/partido" element={<MatchPredictionPage />} />
          <Route path="/odds" element={<Navigate to="/" replace />} />
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
