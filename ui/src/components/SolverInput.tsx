import React, { useState } from 'react';

interface SolverInputProps {
    onSolve: (problemText: string) => void;
    isLoading: boolean;
}

export const SolverInput: React.FC<SolverInputProps> = ({ onSolve, isLoading }) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onSolve(input);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-sm border border-slate-100">
            <h2 className="text-xl font-semibold text-slate-800 mb-4">Resolver Problema</h2>
            <form onSubmit={handleSubmit} className="flex gap-3">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Escribe tu problema matemático (ej: 2x + 4 = 10)..."
                    className="flex-1 px-4 py-3 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                    disabled={isLoading}
                />
                <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="px-6 py-3 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {isLoading ? 'Analizando...' : 'Resolver'}
                </button>
            </form>
        </div>
    );
};
