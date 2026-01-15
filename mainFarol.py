from Projeto.simulador.simuladorFarol import simuladorFarol

if __name__ == "__main__":
    sim = simuladorFarol()


    sim.treinar(total_episodios=5000)

    input("Pressiona Enter para sair...")