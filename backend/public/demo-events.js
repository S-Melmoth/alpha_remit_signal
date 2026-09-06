// Deterministic browser simulation. Production workers live in behaviour/.
export const DEMO_NOW = '2026-09-04T10:00:00.000Z';
export function weekKey(iso) {
  const d = new Date(new Date(iso).getTime() + 3 * 3600000); // Europe/Moscow
  d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7);
  return d.toISOString().slice(0, 10);
}
export function freshEvents() {
  return { now: DEMO_NOW, income: true, inWindow: true, global: [], completed: [], schedules: [], log: [], sequence: 0 };
}
export function weeklyCount(state) { return state.completed.filter(p => weekKey(p.sent_at) === weekKey(state.now)).length; }
export function dispatch(state, kind, alreadyTransferred = false) {
  const texts = {
    behaviour: 'Пришло время отправить деньги за границу, сейчас отличное время.',
    history: 'Модель на исторических данных прогнозирует подходящий момент для перевода. Посмотрите условия в приложении.',
    news: 'Новостная модель нашла подходящий момент для перевода за границу. Проверьте условия перед отправкой.',
  };
  for (const type of kind === 'all' ? ['history', 'news'] : kind === 'behaviour' ? [] : [kind]) {
    state.global.push({ id: `signal-${++state.sequence}`, type, text: texts[type], created_at: state.now, source: 'synthetic_open_data' });
  }
  const behaviour = { id: `pattern-${weekKey(state.now)}`, type: 'behaviour', text: texts.behaviour };
  const candidates = [behaviour, ...state.global].filter(p => p.type !== 'behaviour' || (state.income && state.inWindow && !alreadyTransferred))
    .filter(p => !state.completed.some(c => c.source_id === p.id))
    .sort((a, b) => ['behaviour', 'history', 'news'].indexOf(a.type) - ['behaviour', 'history', 'news'].indexOf(b.type));
  // Other feature demos have no personal behaviour signal unless comparing priorities.
  const applicable = candidates.filter(p => kind === 'all' || kind === 'behaviour' || p.type !== 'behaviour');
  if (weeklyCount(state) >= 2) return { reason: 'Лимит недели: уже отправлено 2 пуша. Следующая отправка возможна на новой неделе.' };
  if (!applicable.length) return { reason: !state.income ? 'Пуш не отправлен: нет поступления, которое подтверждает паттерн.' : !state.inWindow ? 'Пуш не отправлен: сейчас вне временного окна.' : alreadyTransferred ? 'Пуш не отправлен: перевод на этой неделе уже выполнен.' : 'Пуш по этому паттерну уже отправлен. Повтор исключён.' };
  const candidate = applicable[0];
  const push = { id: `push-${++state.sequence}`, user_id: 'demo-user', source_id: candidate.id, type: candidate.type, text: candidate.text, sent_at: state.now };
  state.completed.push(push);
  return { push, reason: `Отправлен ${candidate.type === 'behaviour' ? 'поведенческий' : candidate.type === 'history' ? 'прогнозный' : 'новостной'} пуш. Использовано ${weeklyCount(state)} из 2 слотов недели.` };
}
