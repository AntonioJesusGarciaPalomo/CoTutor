import React from 'react';
import { CheckCircleIcon, LightBulbIcon } from '@heroicons/react/24/outline';

export interface SolutionData {
    final_answer: string;
    steps: string[];
    verification: string;
    problem_text: string;
}

interface SolutionViewProps {
    solution: SolutionData;
    onStartTutor: () => void;
    isStartingSession: boolean;
}

export const SolutionView: React.FC<SolutionViewProps> = ({ solution, onStartTutor, isStartingSession }) => {
    return (
        <div className="w-full max-w-2xl mx-auto mt-6 bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden">
            <div className="p-6 border-b border-slate-100 bg-slate-50">
                <h3 className="text-lg font-semibold text-slate-800">Solución</h3>
                <p className="text-slate-500 text-sm mt-1">{solution.problem_text}</p>
            </div>

            <div className="p-6 space-y-6">
                {/* Answer */}
                <div className="flex items-start gap-3 p-4 bg-green-50 rounded-lg border border-green-100">
                    <CheckCircleIcon className="w-6 h-6 text-green-600 mt-0.5" />
                    <div>
                        <h4 className="font-medium text-green-900">Respuesta Final</h4>
                        <p className="text-green-800 text-lg font-serif mt-1">{solution.final_answer}</p>
                    </div>
                </div>

                {/* Steps */}
                {solution.steps.length > 0 && (
                    <div>
                        <h4 className="font-medium text-slate-700 mb-3">Pasos paso a paso</h4>
                        <div className="space-y-3">
                            {solution.steps.map((step, idx) => (
                                <div key={idx} className="flex gap-3 text-slate-600">
                                    <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full bg-slate-100 text-xs font-medium text-slate-500">
                                        {idx + 1}
                                    </span>
                                    <p>{step}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Verification */}
                {solution.verification && (
                    <div className="bg-slate-50 p-4 rounded-lg text-sm text-slate-600">
                        <span className="font-medium text-slate-700">Verificación: </span>
                        {solution.verification}
                    </div>
                )}

                {/* Action */}
                <div className="pt-4 border-t border-slate-100 flex justify-end">
                    <button
                        onClick={onStartTutor}
                        disabled={isStartingSession}
                        className="flex items-center gap-2 px-5 py-2.5 bg-sky-100 text-sky-700 rounded-lg hover:bg-sky-200 transition-colors font-medium disabled:opacity-50"
                    >
                        <LightBulbIcon className="w-5 h-5" />
                        {isStartingSession ? 'Iniciando sesión...' : 'Necesito ayuda del Tutor'}
                    </button>
                </div>
            </div>
        </div>
    );
};
