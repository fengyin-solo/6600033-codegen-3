import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface MCScenario {
  id: string
  name: string
  description: string
  params: Record<string, number>
  category: string
}

export interface MCResult {
  scenario: string
  iterations: number
  estimate: number
  trueValue?: number
  error?: number
  samples: number[]
  convergence: number[]
}

export interface HypTestResult {
  testType: string
  statistic: number
  pValue: number
  significant: boolean
  alpha: number
  df?: number
}

export interface SavedExperiment {
  id: string
  name: string
  savedAt: number
  scenario: MCScenario
  iterations: number
  result: MCResult
}

export interface ResultDiff {
  estimateDiff: number
  estimateChangePct: number
  errorDiff?: number
  errorChangePct?: number
  isImproved: boolean
}

function normalRandom(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

function runMC(scenario: MCScenario, n: number): MCResult {
  const samples: number[] = []
  const convergence: number[] = []

  if (scenario.id === 'pi') {
    let inside = 0
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1
      if (x * x + y * y <= 1) inside++
      samples.push(x * x + y * y <= 1 ? 1 : 0)
      convergence.push((inside / (i + 1)) * 4)
    }
    const estimate = (inside / n) * 4
    return { scenario: 'pi', iterations: n, estimate, trueValue: Math.PI, error: Math.abs(estimate - Math.PI), samples, convergence }
  }
  if (scenario.id === 'brownian') {
    let pos = 0
    const dt = scenario.params.dt || 0.01
    for (let i = 0; i < n; i++) { pos += normalRandom() * Math.sqrt(dt); samples.push(pos) }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'brownian', iterations: n, estimate: pos, samples, convergence }
  }
  if (scenario.id === 'option') {
    const { S0 = 100, K = 105, r = 0.05, sigma = 0.2, T = 1 } = scenario.params
    let payoffSum = 0
    for (let i = 0; i < n; i++) {
      const ST = S0 * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * normalRandom())
      const p = Math.max(ST - K, 0); payoffSum += p; samples.push(p)
      if ((i + 1) % 50 === 0) convergence.push((payoffSum / (i + 1)) * Math.exp(-r * T))
    }
    return { scenario: 'option', iterations: n, estimate: (payoffSum / n) * Math.exp(-r * T), samples, convergence }
  }
  if (scenario.id === 'random_walk') {
    let pos = 0
    for (let i = 0; i < n; i++) { pos += Math.random() > 0.5 ? 1 : -1; samples.push(pos) }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'random_walk', iterations: n, estimate: pos, samples, convergence }
  }
  if (scenario.id === 'diffusion') {
    const { D = 1, dt = 0.01 } = scenario.params
    let x = 0, y = 0
    for (let i = 0; i < n; i++) {
      x += normalRandom() * Math.sqrt(2 * D * dt); y += normalRandom() * Math.sqrt(2 * D * dt)
      samples.push(Math.sqrt(x * x + y * y))
    }
    convergence.push(...samples.slice(0, 200))
    return { scenario: 'diffusion', iterations: n, estimate: Math.sqrt(x * x + y * y), samples, convergence }
  }
  // gambler
  const { p = 0.45, bankroll = 50, goal = 100 } = scenario.params
  let ruinCount = 0
  for (let i = 0; i < n; i++) {
    let money = bankroll
    let steps = 0
    while (money > 0 && money < goal && steps < 10000) { money += Math.random() < p ? 1 : -1; steps++ }
    if (money <= 0) ruinCount++
    samples.push(money <= 0 ? 0 : 1)
    convergence.push(ruinCount / (i + 1))
  }
  return { scenario: 'gambler', iterations: n, estimate: ruinCount / n, samples, convergence }
}

export const SCENARIOS: MCScenario[] = [
  { id: 'pi', name: '圆周率π估算', description: '随机投点估算π值，观察收敛过程', params: {}, category: '基础' },
  { id: 'brownian', name: '布朗运动模拟', description: '粒子热运动随机路径模拟', params: { dt: 0.01 }, category: '物理' },
  { id: 'option', name: '欧式期权定价', description: 'Black-Scholes期权价格蒙特卡洛估算', params: { S0: 100, K: 105, r: 0.05, sigma: 0.2, T: 1 }, category: '金融' },
  { id: 'random_walk', name: '随机游走', description: '一维离散随机游走轨迹模拟', params: {}, category: '基础' },
  { id: 'diffusion', name: '粒子扩散', description: '二维粒子随机扩散位移分析', params: { D: 1, dt: 0.01 }, category: '物理' },
  { id: 'gambler', name: '赌徒破产', description: '不利赌局下资金耗尽概率估算', params: { p: 0.45, bankroll: 50, goal: 100 }, category: '概率' }
]

const STORAGE_KEY = 'mc_saved_experiments'

function loadFromStorage(): SavedExperiment[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveToStorage(list: SavedExperiment[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
  } catch {
  }
}

export const useMCStore = defineStore('mc', () => {
  const currentScenario = ref<MCScenario>(SCENARIOS[0])
  const iterations = ref(1000)
  const result = ref<MCResult | null>(null)
  const testResult = ref<HypTestResult | null>(null)
  const isRunning = ref(false)
  const savedExperiments = ref<SavedExperiment[]>(loadFromStorage())
  const compareIds = ref<Set<string>>(new Set())
  const replaySourceId = ref<string | null>(null)

  function runSimulation() {
    isRunning.value = true
    setTimeout(() => { result.value = runMC(currentScenario.value, iterations.value); isRunning.value = false }, 10)
  }

  function runTest(g1: number[], g2: number[]) {
    const n1 = g1.length, n2 = g2.length
    const m1 = g1.reduce((a, b) => a + b, 0) / n1
    const m2 = g2.reduce((a, b) => a + b, 0) / n2
    const v1 = g1.reduce((s, x) => s + (x - m1) ** 2, 0) / (n1 - 1)
    const v2 = g2.reduce((s, x) => s + (x - m2) ** 2, 0) / (n2 - 1)
    const se = Math.sqrt(v1 / n1 + v2 / n2)
    const t = (m1 - m2) / se
    const df = Math.round((v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
    const pValue = 2 * (1 - Math.min(0.9999, Math.abs(t) / (Math.abs(t) + Math.sqrt(df))))
    testResult.value = { testType: 'Welch T检验', statistic: Math.round(t * 1000) / 1000, pValue: Math.round(pValue * 10000) / 10000, significant: pValue < 0.05, alpha: 0.05, df }
  }

  function setScenario(s: MCScenario) { currentScenario.value = s; result.value = null; replaySourceId.value = null }

  function saveExperiment(name?: string) {
    if (!result.value) return
    const exp: SavedExperiment = {
      id: 'exp_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7),
      name: name || `${currentScenario.value.name} - ${iterations.value}次 @ ${new Date().toLocaleTimeString()}`,
      savedAt: Date.now(),
      scenario: JSON.parse(JSON.stringify(currentScenario.value)),
      iterations: iterations.value,
      result: JSON.parse(JSON.stringify(result.value))
    }
    savedExperiments.value.unshift(exp)
    saveToStorage(savedExperiments.value)
    return exp
  }

  function deleteExperiment(id: string) {
    savedExperiments.value = savedExperiments.value.filter(e => e.id !== id)
    compareIds.value.delete(id)
    saveToStorage(savedExperiments.value)
  }

  function loadExperiment(id: string) {
    const exp = savedExperiments.value.find(e => e.id === id)
    if (!exp) return
    currentScenario.value = exp.scenario
    iterations.value = exp.iterations
    result.value = exp.result
  }

  function replayExperiment(id: string) {
    const exp = savedExperiments.value.find(e => e.id === id)
    if (!exp) return
    replaySourceId.value = id
    if (!compareIds.value.has(id)) {
      compareIds.value.add(id)
      compareIds.value = new Set(compareIds.value)
    }
    currentScenario.value = exp.scenario
    iterations.value = exp.iterations
    runSimulation()
  }

  function toggleCompare(id: string) {
    if (compareIds.value.has(id)) {
      compareIds.value.delete(id)
    } else {
      compareIds.value.add(id)
    }
    compareIds.value = new Set(compareIds.value)
  }

  function clearCompare() {
    compareIds.value.clear()
    compareIds.value = new Set(compareIds.value)
    replaySourceId.value = null
  }

  function clearReplaySource() {
    replaySourceId.value = null
  }

  const convergenceData = computed(() => {
    if (!result.value) return [] as [number, number][]
    return result.value.convergence.slice(0, 200).map((v, i): [number, number] => [i, Math.round(v * 100000) / 100000])
  })

  const compareConvergenceList = computed(() => {
    const colors = ['#f97316', '#10b981', '#f43f5e', '#8b5cf6', '#eab308', '#06b6d4']
    return Array.from(compareIds.value).map((id, idx) => {
      const exp = savedExperiments.value.find(e => e.id === id)
      if (!exp) return null
      return {
        id,
        name: exp.name,
        color: colors[idx % colors.length],
        data: exp.result.convergence.slice(0, 200).map((v, i): [number, number] => [i, Math.round(v * 100000) / 100000])
      }
    }).filter(Boolean) as { id: string; name: string; color: string; data: [number, number][] }[]
  })

  const histogramData = computed(() => {
    if (!result.value) return { xAxis: [] as number[], data: [] as number[] }
    const s = result.value.samples.slice(0, 1000)
    const mn = Math.min(...s), mx = Math.max(...s)
    const bins = 20, bs = (mx - mn) / bins || 1
    const counts = new Array(bins).fill(0)
    s.forEach(v => { counts[Math.min(bins - 1, Math.floor((v - mn) / bs))]++ })
    return { xAxis: Array.from({ length: bins }, (_, i) => Math.round((mn + i * bs) * 100) / 100), data: counts }
  })

  const replaySource = computed(() => {
    if (!replaySourceId.value) return null
    return savedExperiments.value.find(e => e.id === replaySourceId.value) || null
  })

  const resultDiff = computed((): ResultDiff | null => {
    if (!result.value || !replaySource.value) return null
    const oldR = replaySource.value.result
    const newR = result.value
    const estimateDiff = newR.estimate - oldR.estimate
    const estimateChangePct = oldR.estimate !== 0 ? (estimateDiff / Math.abs(oldR.estimate)) * 100 : 0
    let errorDiff: number | undefined
    let errorChangePct: number | undefined
    let isImproved = false
    if (oldR.error !== undefined && newR.error !== undefined) {
      errorDiff = newR.error - oldR.error
      errorChangePct = oldR.error !== 0 ? (errorDiff / oldR.error) * 100 : 0
      isImproved = newR.error < oldR.error
    }
    return { estimateDiff, estimateChangePct, errorDiff, errorChangePct, isImproved }
  })

  return {
    currentScenario, iterations, result, testResult, isRunning,
    savedExperiments, compareIds, replaySourceId, replaySource, resultDiff,
    convergenceData, compareConvergenceList, histogramData,
    runSimulation, runTest, setScenario,
    saveExperiment, deleteExperiment, loadExperiment, replayExperiment, toggleCompare, clearCompare, clearReplaySource
  }
})
