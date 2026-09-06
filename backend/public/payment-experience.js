import { COUNTRIES, PURPOSES, calculateTransfer, formatMoney, validPhone } from './transfer.js';
import { DEMO_NOW, freshEvents, dispatch, weeklyCount, weekKey } from './demo-events.js';

export const featureKey = new URLSearchParams(location.search).get('feature');
export const hasFeature = ['behaviour','news','recommendation','prefill','scheduler'].includes(featureKey);
export const demoBalance = 83500.35;
export function seedTransfers() {
  return [14, 7].map((days, i) => ({ id:`history-${i}`, date:new Date(new Date(DEMO_NOW).getTime()-days*86400000).toISOString(), country:'Таджикистан', bank:'Душанбе Сити Банк', phone:'+7 926 577-51-28', recipientName:'Ш. А. Д.', purpose:PURPOSES[1][0], message:'', synthetic_history:true, ...calculateTransfer('16000',COUNTRIES.find(c=>c.name==='Таджикистан'),demoBalance) }));
}
export function createExperience(api) {
  const storageKey = `payments-events-v2-${hasFeature ? featureKey : 'payments'}`;
  let state = freshEvents();
  try { const saved=JSON.parse(localStorage.getItem(storageKey)); if(saved && Array.isArray(saved.completed) && Array.isArray(saved.schedules))state=saved; } catch {}
  let desktop=false, visiblePush=null, label='Платежи', message='Приложение готово. Нажимайте на плитки, вводите данные и отправляйте демопереводы.', path=['phone'];
  let form=null, selected=null, lastScreen='', lastModal='';
  const esc=api.escapeHTML;
  const save=()=>{try{localStorage.setItem(storageKey,JSON.stringify(state));}catch{api.toast('Хранилище недоступно. Данные сохранятся до закрытия страницы.');}};
  function log(nextLabel, nextMessage, nextPath) { label=nextLabel;message=nextMessage;path=nextPath;state.log.push(nextMessage);state.log=state.log.slice(-16);save(); }
  function historic(){return api.get().transactions.filter(t=>t.synthetic_history).at(-1);}
  function recurring(t){return api.get().transactions.filter(p=>p.country===t.country && p.bank===t.bank && p.phone===t.phone && p.total===t.total).length>=2;}
  function shortcut(){
    const t=historic(); if(!hasFeature || !t)return '';
    const weeks=Math.max(1,Math.round((new Date(state.now)-new Date(t.date))/604800000));
    return `<button class="experience-shortcut" data-action="repeat-payment"><img src="/assets/bank-dushanbe.png" alt=""><span><strong>Повторить платёж ${formatMoney(t.total,0)} ₽</strong><small>${esc(t.recipientName)} · как ${weeks} ${weeks===1?'неделю':weeks<5?'недели':'недель'} назад</small></span><b>↗</b></button>`;
  }
  function snapshot(){
    const app=api.get();
    const transfers=app.transactions.map(t=>({id:t.id,user_id:'demo-user',kind:'cross_border',amount_minor:Math.round(t.total*100),recipient:t.recipientName||'Ш. А. Д.',phone:t.phone,bank:t.bank,country:t.country,purpose:t.purpose,created_at:t.date,synthetic_history:!!t.synthetic_history}));
    if(state.income)transfers.unshift({id:`income-${weekKey(state.now)}`,user_id:'demo-user',kind:'income',amount_minor:8000000,created_at:state.now});
    return {label,message,path,log:state.log,income:state.income,inWindow:state.inWindow,weekPushes:weeklyCount(state),tables:{
      transfers,
      behaviour_push:[{user_id:'demo-user',window_start:state.now,window_end:new Date(new Date(state.now).getTime()+7200000).toISOString(),in_window:state.inWindow,income_required:true,income_found:state.income,typical_amount_minor:1600000}],
      global_notifications:state.global,completed_push:state.completed,scheduled_cis_payments:state.schedules,
      open_data:[{dataset:'synthetic FX / news fixtures',currency:'TJS',real_market_data:false}],
      ml_news:state.global.filter(s=>s.type==='news'),ml_history:state.global.filter(s=>s.type==='history'),
      pusher:[{priority:'behaviour > history > news',weekly_limit:2,sent_this_week:weeklyCount(state),transport:'browser_demo'}],
      history_api:historic()?[{method:'GET',path:'/history',user_id:'demo-user',payment_id:historic().id,prefill:{country:historic().country,bank:historic().bank,phone:historic().phone,recipient:historic().recipientName,amount:historic().total,purpose:historic().purpose}}]:[],
      scheduler_api:[{operations:'create / list / get / update / cancel',storage:'browser_demo',confirmation_required:true}],
      phone:[{screen:app.screen,modal:app.modal,amount:app.draft.amount,recipient:app.draft.phone,desktop}],
    }};
  }
  function notify(){if(hasFeature && parent!==window)parent.postMessage({source:'payments-experience',state:snapshot()},location.origin);}
  function afterRender(){
    const current=api.get();
    if(current.screen!==lastScreen || current.modal!==lastModal){
      lastScreen=current.screen;lastModal=current.modal;
      const names={payments:'Платежи',transfer:'Форма перевода',confirmation:'Проверка перевода',success:'Перевод выполнен',history:'История переводов',home:'Главный экран','schedule-form':'Настройка расписания','schedule-review':'Проверка расписания',schedules:'Постоянные платежи','schedule-details':'Детали расписания',methods:'Способ перевода'};
      log(names[current.screen]||'Приложение', current.modal ? `Открыт выбор: ${{countries:'страны',banks:'банка',purpose:'назначения',phone:'получателя'}[current.modal]||current.modal}.` : `Открыт экран «${names[current.screen]||current.screen}».`,current.screen.startsWith('schedule')?['phone','scheduler_api','scheduled_cis_payments']:['phone']);
    }
    const target=document.querySelector('.payments-content') || (current.screen==='home' && document.querySelector('.screen-body'));
    if(target)target.insertAdjacentHTML('afterbegin',shortcut());
    const offer=document.querySelector('.schedule-offer');
    if(offer && hasFeature && !recurring(current.activeTransaction))offer.hidden=true;
    if(desktop){
      document.querySelector('#app > .screen')?.setAttribute('inert','');
      document.querySelector('#app').insertAdjacentHTML('beforeend',`<section class="push-desktop" aria-label="Экран телефона"><img class="push-wallpaper" src="/assets/push-home-screen.png" alt="Библиотека приложений на вашем iPhone">${visiblePush?`<button class="push-card" data-action="open-push"><span class="push-app-icon">А</span><span class="push-copy"><span class="push-title"><strong>Альфа-Банк</strong><small>сейчас</small></span><p>${esc(visiblePush.text)}</p><em>Нажмите, чтобы открыть перевод</em></span></button><button class="push-close" data-action="dismiss-push" aria-label="Скрыть уведомление">×</button>`:''}<button class="push-return" data-action="open-bank">Открыть Альфа-Банк</button></section>`);
    }
    notify();
  }
  function repeat(){
    desktop=false;api.prefill(historic());
    log('Реквизиты предзаполнены','Behaviour /history вернул платёж. Можно изменить сумму, получателя, банк и назначение перед подтверждением.',['behaviour_push','transfers','history_api','phone']);notify();
  }
  function closeDesktop(){desktop=false;api.route('payments');}
  function command(action, data={}){
    if(api.get().sending || document.querySelector('.phone-confirm[open]'))return;
    if(action==='income'||action==='window'){
      state[action==='income'?'income':'inWindow']=!!data.enabled;
      log('Условия паттерна изменены','Проверьте условия и вызовите Pusher повторно.',['transfers','behaviour_push']);save();notify();return;
    }
    if(action==='push'||action==='all-signals'){
      const kind=action==='all-signals'?'all':{behaviour:'behaviour',news:'news',recommendation:'history'}[featureKey]||'behaviour';
      const transferred=api.get().transactions.some(t=>!t.synthetic_history && weekKey(t.date)===weekKey(state.now));
      const result=dispatch(state,kind,transferred);
      if(result.push){visiblePush=result.push;desktop=true;api.render();}
      log(result.push?'Новое уведомление':'Отправка остановлена',result.reason,result.push?result.push.type==='behaviour'?['transfers','behaviour_push','pusher','completed_push','phone']:['open_data',result.push.type==='news'?'ml_news':'ml_history','global_notifications','pusher','completed_push','phone']:['pusher','completed_push']);notify();return;
    }
    if(action==='next-week'){
      state.now=new Date(new Date(state.now).getTime()+604800000).toISOString();state.income=false;visiblePush=null;
      log('Новая неделя','Лимит обновлён. Для поведенческого пуша нужно новое поступление.',['transfers','behaviour_push','completed_push']);api.render();return;
    }
    if(action==='reset'){state=freshEvents();desktop=false;visiblePush=null;save();api.reset();log('Демо сброшено','Восстановлены исходные переводы. Пуши и расписания очищены.',['phone']);notify();return;}
    if(action==='desktop'){desktop=true;api.render();return;}
    if(action==='schedules'){desktop=false;api.route('schedules');return;}
    if(action==='app')closeDesktop();
  }
  function beginSchedule(t, existing=null){
    const now=hasFeature?state.now:new Date().toISOString();
    form=existing?{...existing}:{amount:t.total,recipient:t.recipientName||'Ш. А. Д.',phone:t.phone,country:t.country,purpose:t.purpose,date:new Date(new Date(now).getTime()+604800000).toISOString().slice(0,10),time:'13:00',frequency:'weekly',timezone:'Europe/Moscow'};
    selected=existing?.id||null;api.route('schedule-form');
  }
  function scheduleSummary(s){return `<dl>${api.detail('Получатель',s.recipient)}${api.detail('Телефон',s.phone)}${api.detail('Страна и банк',`${s.country} · ${COUNTRIES.find(c=>c.name===s.country)?.bank||''}`)}${api.detail('Сумма',formatMoney(s.amount)+' ₽')}${api.detail('Первый платёж',`${s.date} в ${s.time} · Москва`)}${api.detail('Повторение',s.frequency==='weekly'?'Каждую неделю':'Каждый месяц')}${api.detail('Назначение',s.purpose)}</dl>`;}
  function customScreen(name){
    const shell=(title,body,back='open-schedules')=>`<div class="screen">${api.topbar(title,back)}<div class="screen-body page-pad">${body}</div>${api.nav()}</div>`;
    if(name==='schedule-form' && form){
      const field=(label,name,type='text',extra='')=>`<label>${label}<input name="${name}" type="${type}" value="${esc(form[name])}" ${extra} required></label>`;
      return shell(selected?'Изменить платёж':'Постоянный платёж',`<form id="schedule-form" class="schedule-form">${field('Сумма, ₽','amount','number','min="100" max="100000" step="0.01"')}<div class="fields-row">${field('Дата первого платежа','date','date',`min="${(hasFeature?state.now:new Date().toISOString()).slice(0,10)}"`)}${field('Время по Москве','time','time')}</div><label>Повторять<select name="frequency"><option value="weekly" ${form.frequency==='weekly'?'selected':''}>Каждую неделю</option><option value="monthly" ${form.frequency==='monthly'?'selected':''}>Каждый месяц</option></select></label>${field('Получатель','recipient','text','maxlength="80"')}${field('Номер телефона','phone','tel','maxlength="20"')}<label>Страна и банк<select name="country">${COUNTRIES.filter(c=>c.available).map(c=>`<option value="${esc(c.name)}" ${form.country===c.name?'selected':''}>${esc(c.name)} · ${esc(c.bank)}</option>`).join('')}</select></label><label>Назначение<select name="purpose">${PURPOSES.map(p=>`<option ${form.purpose===p[0]?'selected':''}>${esc(p[0])}</option>`).join('')}</select></label><p class="inline-note">Расписание по московскому времени. Если в месяце нет выбранного числа, платёж приходится на последний день месяца.</p><p class="amount-error" id="schedule-error" role="alert"></p><button type="submit" class="primary">Проверить данные</button></form>`);
    }
    if(name==='schedule-review' && form)return shell('Проверьте расписание',`${scheduleSummary(form)}<button class="primary" data-action="confirm-schedule">${selected?'Сохранить изменения':'Создать постоянный платёж'}</button><button class="secondary" data-action="edit-schedule-form">Изменить данные</button>`,'edit-schedule-form');
    if(name==='schedules')return shell('Мои постоянные платежи',`<p class="demo-inline">Демонстрационные расписания. Реальные списания не выполняются.</p>${state.schedules.length?state.schedules.map(s=>`<button class="schedule-item" data-action="schedule-details" data-id="${esc(s.id)}"><span><strong>${formatMoney(s.amount,0)} ₽ · ${esc(s.recipient)}</strong><small>${esc(s.date)} · ${esc(s.time)} МСК<br>${s.frequency==='weekly'?'Еженедельно':'Ежемесячно'} · ${s.status==='active'?'Активен':'Отменён'}</small></span>›</button>`).join(''):'<div class="empty-state"><h3>Постоянных платежей пока нет</h3><p>После перевода можно настроить дату и время повторения.</p><button class="secondary" data-action="open-bank">К платежам</button></div>'}`,'open-bank');
    if(name==='schedule-details'){
      const s=state.schedules.find(s=>s.id===selected);if(!s)return shell('Платёж не найден','<button class="secondary" data-action="open-schedules">К списку</button>');
      return shell('Постоянный платёж',`${scheduleSummary(s)}<p class="inline-note">${s.status==='active'?'Активен':'Отменён — списаний по этому расписанию не будет'}</p>${s.status==='active'?'<button class="primary" data-action="edit-schedule">Изменить</button><button class="secondary cancel-schedule" data-action="cancel-schedule">Отменить постоянный платёж</button>':''}`);
    }
    return null;
  }
  function confirmDialog(cancel=false){
    const dialog=document.createElement('dialog');dialog.className='phone-confirm';
    dialog.innerHTML=`<h2>${cancel?'Отменить постоянный платёж?':selected?'Подтвердить изменения?':'Создать постоянный платёж?'}</h2><p>${cancel?'Будущие платежи по этому расписанию выполняться не будут.':'Проверьте сумму и расписание. Постоянный платёж можно отменить в любой момент в разделе «Мои платежи».'}</p><button class="primary" id="schedule-confirm-yes">${cancel?'Да, отменить':'Подтверждаю'}</button><button class="secondary" id="schedule-confirm-no">${cancel?'Оставить платёж':'Вернуться к проверке'}</button>`;
    document.body.append(dialog);dialog.showModal();
    dialog.addEventListener('close',()=>dialog.remove());
    dialog.querySelector('#schedule-confirm-no').onclick=()=>dialog.close();
    dialog.querySelector('#schedule-confirm-yes').onclick=()=>{
      if(cancel){const s=state.schedules.find(s=>s.id===selected);s.status='cancelled';s.updated_at=state.now;}
      else{
        const record={...form,id:selected||crypto.randomUUID(),user_id:'demo-user',amount_minor:Math.round(form.amount*100),bank:COUNTRIES.find(c=>c.name===form.country).bank,status:'active',confirmed:true,updated_at:state.now};
        const index=state.schedules.findIndex(s=>s.id===record.id);
        if(index>=0)state.schedules[index]=record;else state.schedules.push(record);selected=record.id;
      }
      save();dialog.close();api.route('schedule-details');
      log(cancel?'Платёж отменён':'Расписание сохранено',cancel?'Payment Scheduler отменил будущие платежи.':'После подтверждения сохранена запись scheduled_cis_payments.',['phone','scheduler_api','scheduled_cis_payments']);notify();
    };
  }
  document.addEventListener('submit',e=>{
    if(e.target.id!=='schedule-form')return;e.preventDefault();
    const fields=Object.fromEntries(new FormData(e.target));
    if(!validPhone(fields.phone)){document.querySelector('#schedule-error').textContent='Проверьте номер телефона получателя.';return;}
    if(!fields.recipient.trim()){document.querySelector('#schedule-error').textContent='Укажите получателя.';return;}
    const now=hasFeature?state.now:new Date().toISOString();
    if(new Date(`${fields.date}T${fields.time}:00+03:00`)<=new Date(now)){document.querySelector('#schedule-error').textContent='Выберите дату и время в будущем.';return;}
    form={...form,...fields,amount:Number(fields.amount),recipient:fields.recipient.trim()};api.route('schedule-review');
  });
  // Keep edits when a user returns from the review screen.
  document.addEventListener('input',e=>{if(e.target.closest('#schedule-form') && e.target.name)form[e.target.name]=e.target.value;});
  window.addEventListener('message',e=>{if(hasFeature && e.origin===location.origin && e.source===parent && e.data?.source==='feature-host')command(e.data.action,e.data);});
  function action(name,button){
    const handlers={
      'repeat-payment':repeat,'open-bank':closeDesktop,'open-schedules':()=>api.route('schedules'),'my-payments':()=>api.route('schedules'),
      'open-push':()=>{visiblePush=null;repeat();log('Пуш открыт','Переход из уведомления в предзаполненную форму перевода.',['completed_push','history_api','phone']);notify();},
      'dismiss-push':()=>{visiblePush=null;api.render();},
      'schedule-demo':()=>beginSchedule(api.get().activeTransaction),
      'schedule-details':()=>{selected=button.dataset.id;api.route('schedule-details');},
      'edit-schedule':()=>{const s=state.schedules.find(s=>s.id===selected);beginSchedule(s,s);},
      'edit-schedule-form':()=>api.route('schedule-form'),
      'confirm-schedule':()=>confirmDialog(),'cancel-schedule':()=>confirmDialog(true),
    };
    if(!handlers[name])return false;handlers[name]();return true;
  }
  return {afterRender,customScreen,action,notify,resetTransfers:api.reset,now:()=>hasFeature?state.now:new Date().toISOString(),onTransfer:()=>{log('Перевод выполнен','Новый перевод записан в историю. Проверьте данные в Tech.',['phone','transfers','history_api']);notify();}};
}
