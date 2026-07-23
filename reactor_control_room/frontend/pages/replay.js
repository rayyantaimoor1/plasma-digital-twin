/* Session Replay page - built in a later commit. */
export async function renderReplay(main) {
  main.innerHTML = `
    <div class="page-head"><div>
      <h1>Session Replay</h1>
      <p>Flight-recorder view of saved experiment runs — under construction.</p>
    </div></div>
    <div class="loading"><span class="dot"></span> This page is being built.</div>`;
  return {};
}
