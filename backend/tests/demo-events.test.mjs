import test from 'node:test';
import assert from 'node:assert/strict';
import { freshEvents, dispatch, weeklyCount, weekKey } from '../public/demo-events.js';
test('behaviour needs income and window, excludes a completed transfer',()=>{
  const state=freshEvents();state.income=false;
  assert.equal(dispatch(state,'behaviour').push,undefined);
  state.income=true;state.inWindow=false;
  assert.equal(dispatch(state,'behaviour').push,undefined);
  state.inWindow=true;
  assert.equal(dispatch(state,'behaviour',true).push,undefined);
  assert.equal(state.completed.length,0);
});
test('two weekly slots, dedup and behaviour > history > news',()=>{
  const state=freshEvents();
  assert.equal(dispatch(state,'all').push.type,'behaviour');
  assert.equal(dispatch(state,'behaviour').push.type,'history');
  assert.equal(dispatch(state,'news').push,undefined);
  assert.equal(weeklyCount(state),2);
  state.now='2026-09-11T10:00:00.000Z';state.income=false;
  assert.equal(weeklyCount(state),0);
  assert.equal(dispatch(state,'news').push.type,'news');
  assert.equal(new Set(state.completed.map(p=>p.source_id)).size,state.completed.length);
});
test('weeks start on Monday midnight Moscow',()=>{
  assert.equal(weekKey('2026-09-06T20:59:59Z'),'2026-08-31');
  assert.equal(weekKey('2026-09-06T21:00:00Z'),'2026-09-07');
});
