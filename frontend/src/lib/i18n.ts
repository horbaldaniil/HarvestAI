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
    fields: {
      title: "Мої поля",
      addNew: "Додати поле",
      empty: {
        title: "Поки що немає полів",
        description: "Натисніть «Додати поле», щоб намалювати своє перше поле на мапі.",
        cta: "Намалювати перше поле",
      },
      hint: {
        drawing:
          "Клацайте, щоб додавати вершини. Завершити — кліком по першій точці, подвійним кліком або кнопкою «Завершити».",
        editing: "Перетягніть вершини, потім натисніть «Зберегти» в панелі.",
      },
      name: "Назва поля",
      namePlaceholder: "Наприклад: «Південне поле №3»",
      crop: "Культура",
      year: "Сезон (рік)",
      area: "Площа",
      areaUnit: "га",
      created: "Створено",
      updated: "Оновлено",
      create: {
        title: "Нове поле",
        description: "Заповніть дані поля. Площу буде розраховано автоматично.",
        submit: "Створити поле",
        success: "Поле створено",
        failed: "Не вдалося створити поле",
      },
      edit: {
        title: "Редагування поля",
        description: "Змініть метадані поля.",
        submit: "Зберегти",
        success: "Зміни збережено",
        failed: "Не вдалося зберегти",
        geometryHint:
          "Щоб змінити форму, скористайтесь кнопкою «Редагувати форму» на мапі.",
      },
      delete: {
        title: "Видалити поле?",
        description:
          "Усі історичні дані для поля «{{name}}» буде втрачено. Цю дію не можна скасувати.",
        confirm: "Видалити",
        success: "Поле видалено",
        failed: "Не вдалося видалити",
      },
      crops: {
        wheat: "Пшениця",
        corn: "Кукурудза",
        sunflower: "Соняшник",
      },
      actions: {
        focus: "Показати на мапі",
        edit: "Редагувати",
        editShape: "Редагувати форму",
        delete: "Видалити",
        cancelDrawing: "Скасувати малювання",
        finishDrawing: "Завершити малювання",
      },
    },
    map: {
      baseLayer: {
        osm: "Карта",
        satellite: "Супутник",
      },
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
