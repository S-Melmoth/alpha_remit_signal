import { features } from './feature.js';
const $ = s => document.querySelector(s);
const escape = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const key = location.pathname.split('/').filter(Boolean).at(-1);
const feature = features[key] || features.behaviour;
const frame = $('#payments-frame');
let mode = 'product', snapshot = null, selectedTable = null;
const pushFeature = ['behaviour','news','recommendation'].includes(key);
const title = {behaviour:'Вызвать поведенческий пуш',news:'Создать новостной сигнал',recommendation:'Создать прогноз модели'};
document.title = `${feature.short} · Ближе, чем кажется`;
$('#feature-short').textContent = feature.short;
$('#feature-kicker').textContent = feature.kicker;
$('#feature-title').textContent = feature.title;
$('#feature-lead').textContent = ({behaviour:'Подтвердите поступление, выберите временное окно и вызовите пуш. Откройте его и совершите перевод в приложении.',news:'Создайте синтетический новостной сигнал, получите уведомление и проверьте условия перевода в приложении.',recommendation:'Вызовите прогноз на синтетических Open Data. Нажмите на пуш и пройдите настоящий путь перевода.',prefill:'Нажмите «Повторить платёж» в разделе «Платежи». Реквизиты появятся в той же форме, где можно изменить сумму и отправить перевод.',scheduler:'Повторите знакомый перевод. После отправки задайте расписание, проверьте его и подтвердите создание. Изменение и отмена доступны в «Моих платежах».'})[key];

$('#event-controls').innerHTML = `${pushFeature ? `<div class="event-options">${key==='behaviour'?'<label><input id="income-toggle" type="checkbox" checked> Поступление есть</label><label><input id="window-toggle" type="checkbox" checked> Сейчас в окне пуша</label>':'<span class="tech-hint">ML создаст готовый текст на синтетических Open Data.</span>'}</div><div class="event-actions"><button class="trigger-push" data-command="push">${title[key]}</button><button data-command="all-signals">Проверить приоритеты</button><button data-command="next-week">Следующая неделя</button><button data-command="reset">Сбросить демо</button></div>` : `<p class="tech-hint">Все действия выполняются внутри телефона. Состояние сохраняется при обновлении страницы.</p><div class="event-actions"><button class="trigger-push" data-command="app">Открыть платежи</button><button data-command="schedules">Мои постоянные платежи</button><button data-command="reset">Сбросить демо</button></div>`}`;
const command = (action, extra={}) => frame.contentWindow?.postMessage({source:'feature-host',action,...extra},location.origin);
document.querySelectorAll('[data-command]').forEach(b=>b.addEventListener('click',()=>command(b.dataset.command)));
$('#income-toggle')?.addEventListener('change',e=>command('income',{enabled:e.target.checked}));
$('#window-toggle')?.addEventListener('change',e=>command('window',{enabled:e.target.checked}));
$('#open-app').addEventListener('click',()=>command('app'));
$('#show-desktop').addEventListener('click',()=>command('desktop'));
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;render();}));
$('#picker-popover').innerHTML = Object.entries(features).map(([id,f])=>`<a role="option" aria-selected="${id===key}" href="/features/${id}" class="${id===key?'active':''}">${f.short}</a>`).join('');
$('#feature-picker').addEventListener('click',()=>{const p=$('#picker-popover');p.hidden=!p.hidden;$('#feature-picker').setAttribute('aria-expanded',String(!p.hidden));});
document.addEventListener('click',e=>{if(!e.target.closest('#feature-picker,#picker-popover'))$('#picker-popover').hidden=true;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('#picker-popover').hidden=true;$('#feature-picker').setAttribute('aria-expanded','false');}});
window.addEventListener('message',e=>{
 if(e.origin!==location.origin || e.source!==frame.contentWindow || e.data?.source!=='payments-experience')return;
 snapshot=e.data.state;
 $('#event-status').textContent=snapshot.message;
 $('#stage-step').textContent=snapshot.label;
 if($('#income-toggle'))$('#income-toggle').checked=snapshot.income;
 if($('#window-toggle'))$('#window-toggle').checked=snapshot.inWindow;
 render();if($('#data-dialog').open && selectedTable)renderTable();
});
function render(){
 document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
 const current=snapshot || {label:'Приложение готовится',message:'Начните с кнопок внутри телефона.',tables:{},path:[],log:[],weekPushes:0};
 if(mode==='product'){
 $('#explanation-content').innerHTML=`<div class="product-content"><div class="current-step"><span class="step-number">↗</span><div><h2>${escape(current.label)}</h2><p>${escape(current.message)}</p></div></div><div class="benefit-card"><span>ПОЧЕМУ ЭТО ХОРОШО</span><p>${escape(feature.benefit)}</p></div><div class="metrics"><div class="metric"><b>${current.weekPushes}/2</b><small>пуша на этой неделе</small></div><div class="metric"><b>${current.tables.transfers?.filter(t=>t.kind==='cross_border'&&!t.synthetic_history).length||0}</b><small>переводов в демо</small></div><div class="metric"><b>${current.tables.scheduled_cis_payments?.filter(s=>s.status==='active').length||0}</b><small>активных расписаний</small></div></div></div>`;
 const phone=current.tables.phone?.[0]||{};
 const steps=key==='scheduler'?['Перевод','Расписание','Проверка','Создан']: [pushFeature?'Пуш':'Плитка','Реквизиты','Проверка','Перевод'];
 const stage=key==='scheduler'?({'schedule-form':1,'schedule-review':2,'schedule-details':3,schedules:3}[phone.screen]||0):phone.desktop?0:({transfer:1,confirmation:2,success:3}[phone.screen]||0);
 $('#explanation-content .product-content').insertAdjacentHTML('afterbegin',`<ol class="product-path" aria-label="Путь пользователя">${steps.map((s,i)=>`<li class="${i===stage?'active':''}" ${i===stage?'aria-current="step"':''}>${s}</li>`).join('')}</ol>`);
 } else {
 const nodes=key==='behaviour'?[['transfers','₽'],['behaviour_push','◷'],['pusher','⇢'],['completed_push','✓'],['phone','▯']]:pushFeature?[['open_data','◎'],[key==='news'?'ml_news':'ml_history','✦'],['global_notifications','▤'],['pusher','⇢'],['completed_push','✓'],['phone','▯']]:key==='prefill'?[['behaviour_push','◷'],['transfers','₽'],['history_api','↗'],['phone','▯']]:[['transfers','₽'],['history_api','↗'],['phone','▯'],['scheduler_api','↻'],['scheduled_cis_payments','▦']];
 const labels={pusher:'Pusher',phone:'Приложение',history_api:'Behaviour /history',scheduler_api:'Payment Scheduler',open_data:'Open Data',ml_news:'Новостная ML',ml_history:'Прогнозная ML'};
 $('#explanation-content').innerHTML=`<div class="tech-content"><p class="tech-hint">Живая симуляция в браузере · данные меняются от действий в телефоне.</p><div class="architecture">${nodes.map(([id,icon])=>`<button class="arch-node ${current.path.includes(id)?'active':''}" data-table="${id}"><span class="node-icon">${icon}</span><b>${labels[id]||id}</b><small>${current.tables[id]?.length??0} строк · открыть</small></button>`).join('')}</div><div class="flow-log"><strong>${escape(current.label)}</strong><ol>${current.log.slice(-4).map(s=>`<li>${escape(s)}</li>`).join('')}</ol></div></div>`;
 document.querySelectorAll('[data-table]').forEach(b=>b.addEventListener('click',()=>{selectedTable=b.dataset.table;renderTable();$('#data-dialog').showModal();}));
 }
}
function renderTable(){
 const rows=snapshot?.tables[selectedTable]||[];
 const cols=[...new Set(rows.flatMap(Object.keys))];
 $('#data-title').textContent=selectedTable;
 $('#data-caption').textContent=`Состояние текущего демо: ${rows.length} строк. Денежные суммы amount_minor хранятся в копейках.`;
 $('#data-table').innerHTML=rows.length?`<thead><tr>${cols.map(c=>`<th>${escape(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((row,i)=>`<tr class="${i===rows.length-1?'changed-row':''}">${cols.map(c=>`<td>${escape(typeof row[c]==='object'?JSON.stringify(row[c]):row[c])}</td>`).join('')}</tr>`).join('')}</tbody>`:'<tbody><tr><td class="empty-table">Пока пусто. Вызовите пуш или выполните действие в телефоне — здесь появятся данные.</td></tr></tbody>';
}
$('#data-close').addEventListener('click',()=>$('#data-dialog').close());
$('#data-dialog').addEventListener('click',e=>{if(e.target===$('#data-dialog'))$('#data-dialog').close();});
frame.src=`/payments?embed=1&feature=${encodeURIComponent(key)}`;
render();
