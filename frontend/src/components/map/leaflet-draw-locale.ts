/**
 * Localize leaflet-draw toolbar strings to Ukrainian.
 *
 * Why a function (not a top-level mutation): the draw locale is global state
 * on the L namespace, but we want HMR-friendly idempotent application.
 */
import L from "leaflet";
import "leaflet-draw";

let applied = false;

export function applyUkrainianDrawLocale(): void {
  if (applied) return;
  applied = true;

  // Use any-typed accessor — leaflet-draw types are incomplete.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dl = (L as any).drawLocal;

  dl.draw.toolbar.actions.title = "Скасувати малювання";
  dl.draw.toolbar.actions.text = "Скасувати";
  dl.draw.toolbar.finish.title = "Завершити малювання";
  dl.draw.toolbar.finish.text = "Завершити";
  dl.draw.toolbar.undo.title = "Видалити останню точку";
  dl.draw.toolbar.undo.text = "Видалити точку";
  dl.draw.toolbar.buttons.polygon = "Намалювати поле (полігон)";

  dl.draw.handlers.polygon.tooltip.start = "Клік щоб почати малювати поле.";
  dl.draw.handlers.polygon.tooltip.cont = "Клік щоб додати наступну вершину.";
  dl.draw.handlers.polygon.tooltip.end =
    "Клік по першій точці щоб завершити поле.";

  dl.edit.toolbar.actions.save.title = "Зберегти зміни";
  dl.edit.toolbar.actions.save.text = "Зберегти";
  dl.edit.toolbar.actions.cancel.title = "Скасувати зміни";
  dl.edit.toolbar.actions.cancel.text = "Скасувати";
  dl.edit.toolbar.actions.clearAll.title = "Очистити все";
  dl.edit.toolbar.actions.clearAll.text = "Очистити все";
  dl.edit.toolbar.buttons.edit = "Редагувати форму поля";
  dl.edit.toolbar.buttons.editDisabled = "Немає поля для редагування";
  dl.edit.toolbar.buttons.remove = "Видалити поле";
  dl.edit.toolbar.buttons.removeDisabled = "Немає поля для видалення";

  dl.edit.handlers.edit.tooltip.text =
    "Перетягніть вершини щоб відредагувати форму.";
  dl.edit.handlers.edit.tooltip.subtext =
    "Натисніть 'Скасувати' щоб відкинути зміни.";
}
