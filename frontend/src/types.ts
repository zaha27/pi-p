export interface Meta {
  type: "meta";
  model: string;
  names: Record<number, string>;
  colors: Record<number, string>;
  fps: number;
  total: number;
  width: number;
  height: number;
  device: string;
  track: boolean;
}

export interface ClassStat {
  id: number;
  count: number;
  share: number;
  spark: number[];
}

export type FlowVector = [x: number, y: number, vx: number, vy: number, cls: number];

export interface Metrics {
  type: "metrics";
  time: number;
  frame: number;
  in_frame: number;
  unique: number;
  mean_flow: number;
  inference_ms: number;
  classes: ClassStat[];
  timeline: { seconds: number; series: number[][] };
  flow: FlowVector[];
  saturation: number;
  peak: [number, number, number];
}

export type ServerMessage =
  | Meta
  | Metrics
  | { type: "done"; files: Record<string, string> }
  | { type: "error"; message: string };
