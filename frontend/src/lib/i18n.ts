export type Locale = "es" | "en";

type Dictionary = Record<string, string>;

const messages: Record<Locale, Dictionary> = {
  es: {
    "header.subtitle": "Demo de prediccion 1X2 con probabilidades calibradas",
    "header.activeModel": "Modelo activo",
    "header.noModel": "Sin modelo entrenado",
    "nav.home": "Inicio",
    "nav.datasets": "Datasets",
    "nav.training": "Entrenamiento",
    "nav.predict": "Prediccion",
    "nav.odds": "Comparar Cuotas"
  },
  en: {
    "header.subtitle": "1X2 prediction demo with calibrated probabilities",
    "header.activeModel": "Active model",
    "header.noModel": "No trained model",
    "nav.home": "Home",
    "nav.datasets": "Datasets",
    "nav.training": "Training",
    "nav.predict": "Predict",
    "nav.odds": "Odds Compare"
  }
};

export function t(locale: Locale, key: string): string {
  return messages[locale][key] ?? key;
}
