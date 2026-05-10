import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const uk = {
  translation: {
    app: {
      name: "HarvestAI",
      tagline: "Супутниковий моніторинг та AI-прогноз врожайності",
    },
    common: {
      loading: "Завантаження...",
      save: "Зберегти",
      cancel: "Скасувати",
      delete: "Видалити",
      edit: "Редагувати",
      submit: "Підтвердити",
      back: "Назад",
      retry: "Спробувати знову",
      logout: "Вийти",
    },
    auth: {
      welcome: "Ласкаво просимо до HarvestAI",
      loginTitle: "Вхід",
      registerTitle: "Реєстрація",
      email: "Email",
      password: "Пароль",
      fullName: "Повне ім'я",
      login: "Увійти",
      register: "Зареєструватись",
      noAccount: "Ще не маєте акаунту?",
      hasAccount: "Вже маєте акаунт?",
      passwordHint: "Мінімум 10 символів",
      registerSuccess: "Реєстрація успішна! Тепер увійдіть.",
      loginFailed: "Невірний email або пароль",
      logoutSuccess: "Ви вийшли з системи",
    },
    nav: {
      dashboard: "Дашборд",
      fields: "Поля",
      chat: "AI-помічник",
      reports: "Звіти",
      settings: "Налаштування",
    },
    dashboard: {
      title: "Дашборд",
      totalFields: "Усього полів",
      totalArea: "Загальна площа",
      activeAlerts: "Активні попередження",
      avgNdvi: "Середній NDVI",
    },
  },
};

i18n.use(initReactI18next).init({
  resources: { uk },
  lng: "uk",
  fallbackLng: "uk",
  interpolation: { escapeValue: false },
});

export default i18n;
