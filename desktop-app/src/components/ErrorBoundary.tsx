import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-screen h-screen bg-[#07090e] text-slate-100 flex flex-col items-center justify-center p-8 select-none">
          <div className="max-w-xl w-full bg-[#0d111a] border border-rose-900/60 rounded-2xl p-8 shadow-2xl flex flex-col gap-5">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-rose-500/20 text-rose-400 flex items-center justify-center border border-rose-500/40 shrink-0">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white font-mono">
                  Runtime Component Exception
                </h2>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  An isolated view component encountered an error and was safely contained.
                </p>
              </div>
            </div>

            <div className="bg-black/60 border border-slate-800 rounded-lg p-4 font-mono text-xs text-rose-300 overflow-x-auto max-h-48 whitespace-pre-wrap">
              {this.state.error?.toString() || 'Unknown runtime error'}
            </div>

            <div className="flex justify-end gap-3 pt-2 border-t border-slate-800">
              <button
                onClick={this.handleReset}
                className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-mono text-xs px-5 py-2.5 rounded-lg font-bold transition shadow-lg shadow-emerald-600/30"
              >
                <RotateCcw className="w-4 h-4" />
                Recover View State
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
